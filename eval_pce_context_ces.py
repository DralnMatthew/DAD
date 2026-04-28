""" Evaluate per-step sPCE for the CES task when an external "context set" of source
trajectories is injected into the design network's history.

Four conditions are evaluated:
    (n_source=1, all relevant)
    (n_source=5, all relevant)
    (n_source=1, all irrelevant)
    (n_source=5, all irrelevant)

Relevance of a source trajectory follows the definition in `sample_correlated_batch`:
    - relevant   : source theta = target theta + small Gaussian noise (sigma = 0.1 * range)
    - irrelevant : source theta = target theta shifted away from midpoint by 0.3..0.5 * range

Per-step sPCE / sNMC bounds are computed using the existing EIGStepLoss machinery and
saved into a single .pt payload.
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from attrdictionary import AttrDict

from data import CES
from model.mlp import EncoderNetworkV2, EmitterNetwork, SetEquivariantDesignNetwork
from loss.eig import EIGStepLoss
from utils import create_logger


# Per-component theta ranges used to define "relevant" / "irrelevant" perturbations.
# rho in [0.01, 1.0]; alpha components in [0, 1] (simplex-constrained downstream);
# log_u uses ~99% of the N(1, 3) prior support.
THETA_RANGE_MIN = torch.tensor([0.01, 0.0, 0.0, 0.0, -8.0])
THETA_RANGE_MAX = torch.tensor([1.0,  1.0, 1.0, 1.0, 10.0])


def load_cfg(run_dir: str, override_config_path: str = None):
    """Load the run's hydra config; if missing, compose defaults from ./config."""
    if override_config_path is not None:
        return OmegaConf.load(override_config_path)
    candidate = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.isfile(candidate):
        return OmegaConf.load(candidate)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(repo_root, "config", "ces.yaml")),
        {"data": OmegaConf.load(os.path.join(repo_root, "config", "data", "ces.yaml"))},
        {"model": OmegaConf.load(os.path.join(repo_root, "config", "model", "mlp_ces.yaml"))},
    )
    return cfg


def build_model(cfg):
    encoder = EncoderNetworkV2(
        cfg.data.dim_design, cfg.data.dim_outcome,
        embed_dim=cfg.model.embed_dim, hidden_dim=cfg.model.hidden_dim,
        encoding_dim=cfg.model.encoding_dim,
        hidden_depth=cfg.model.encoder_hidden_depth,
        activation=nn.ReLU(),
        normalization=cfg.model.encoder_normalization,
    )
    emitter = EmitterNetwork(
        cfg.model.encoding_dim, cfg.data.dim_design,
        hidden_dim=cfg.model.hidden_dim,
        hidden_depth=cfg.model.emitter_hidden_depth,
        activation=nn.ReLU(),
    )
    model = SetEquivariantDesignNetwork(
        encoder, emitter,
        cfg.data.dim_design, cfg.data.dim_outcome,
        empty_value=torch.ones(cfg.model.encoding_dim) * 0.01,
    )
    return model


def build_experiment(cfg):
    return CES(
        dim_design=cfg.data.dim_design, dim_outcome=cfg.data.dim_outcome,
        design_scale=cfg.data.design_scale, noise_scale=cfg.data.noise_scale,
    )


@torch.no_grad()
def make_source_thetas(target_theta: torch.Tensor, n_source: int, relevant: bool,
                       correlation_std: float = 0.1) -> torch.Tensor:
    """Build source thetas relative to a given target theta.

    Args:
        target_theta [B, 5]: rho, alpha[3], log_u
        n_source: number of source trajectories per batch element
        relevant: True -> perturb close to target; False -> push away from target
    Returns:
        source_theta [B, n_source, 5] with rho clamped, alpha re-projected to simplex
    """
    B, D = target_theta.shape
    target_exp = target_theta.unsqueeze(1).expand(-1, n_source, -1)  # [B, N, 5]

    range_min = THETA_RANGE_MIN.to(target_theta).view(1, 1, D)
    range_max = THETA_RANGE_MAX.to(target_theta).view(1, 1, D)
    range_span = range_max - range_min

    if relevant:
        noise = torch.randn_like(target_exp) * correlation_std * range_span
        source = target_exp + noise
    else:
        midpoint = (range_max + range_min) / 2.0
        direction = torch.where(target_exp < midpoint, 1.0, -1.0)
        shift_mag = range_span * torch.empty_like(target_exp).uniform_(0.3, 0.5)
        source = target_exp + direction * shift_mag

    source = torch.clamp(source, min=range_min, max=range_max)

    # rho > 0 strictly
    source[..., 0] = torch.clamp(source[..., 0], 0.01, 1.0)
    # alpha on simplex
    source[..., 1:4] = torch.abs(source[..., 1:4]) + 1e-6
    source[..., 1:4] = source[..., 1:4] / source[..., 1:4].sum(dim=-1, keepdim=True)
    return source


@torch.no_grad()
def make_source_trajectories(experiment: CES, target_theta: torch.Tensor,
                             n_source: int, T_src: int, relevant: bool):
    """Generate source trajectories: random baskets + simulated CES outcomes.

    Args:
        target_theta [B, 5]
        n_source: number of source trajectories
        T_src: length of each source trajectory
        relevant: relevance flag
    Returns:
        source_x [B, n_source * T_src, 6]
        source_y [B, n_source * T_src, 1]
        source_theta [B, n_source, 5]
    """
    B = target_theta.shape[0]
    D = target_theta.shape[1]
    device = target_theta.device

    source_theta = make_source_thetas(target_theta, n_source, relevant)  # [B, N, 5]

    basket_dim = experiment.basket_dim
    basket1 = torch.rand(B, n_source, T_src, basket_dim, device=device) * experiment.design_scale
    basket2 = torch.rand(B, n_source, T_src, basket_dim, device=device) * experiment.design_scale
    source_x = torch.cat([basket1, basket2], dim=-1)                                # [B, N, T_src, 6]

    flat_x = source_x.reshape(-1, experiment.dim_design)
    flat_theta = source_theta.unsqueeze(2).expand(-1, -1, T_src, -1).reshape(-1, D)
    flat_y = experiment.forward(flat_x, flat_theta)                                  # [B*N*T_src, 1]
    source_y = flat_y.reshape(B, n_source, T_src, experiment.dim_outcome)

    # Flatten the (n_source, T_src) axes into a single "context set" axis.
    source_x_flat = source_x.reshape(B, n_source * T_src, experiment.dim_design)
    source_y_flat = source_y.reshape(B, n_source * T_src, experiment.dim_outcome)
    return source_x_flat, source_y_flat, source_theta


@torch.no_grad()
def run_trace_with_context(model, experiment, T: int, batch_size: int,
                           context_x: torch.Tensor, context_y: torch.Tensor,
                           target_theta: torch.Tensor):
    """Run T-step adaptive trajectories, prepending `context_x/y` to the encoder history.

    Args:
        context_x [B, N_ctx, D_x]
        context_y [B, N_ctx, D_y]
        target_theta [B, 5]
    Returns:
        target_theta, xi_designs [B, T, D_x], y_outcomes [B, T, D_y]
    """
    model.eval()
    xi_full = torch.empty((batch_size, T, model.dim_x), device=target_theta.device)
    xi_designs = torch.empty((batch_size, T, model.dim_x), device=target_theta.device)
    y_outcomes = torch.empty((batch_size, T, model.dim_y), device=target_theta.device)

    for t in range(T):
        hist_x = torch.cat([context_x, xi_full[:, :t]], dim=1)
        hist_y = torch.cat([context_y, y_outcomes[:, :t]], dim=1)
        xi = model.forward(hist_x, hist_y)
        xi_design = experiment.to_design_space(xi)
        y = experiment(xi_design, target_theta)

        xi_full[:, t] = xi
        xi_designs[:, t] = xi_design
        y_outcomes[:, t] = y

    return target_theta, xi_designs, y_outcomes


@torch.no_grad()
def compute_stepwise_bounds(experiment, theta_0, x, y, L, batch_size):
    """Per-step sPCE/sNMC for a set of completed trajectories. Mirrors compute_EIG_from_history."""
    T = x.shape[1]
    criterion = EIGStepLoss(L, batch_size, experiment.log_likelihood, reduction='none')
    thetas = experiment.sample_theta((L, batch_size))
    thetas = torch.cat([theta_0.unsqueeze(0), thetas], dim=0)

    pce_losses, nmc_losses = [], []
    for t in range(T):
        pce_loss, nmc_loss = criterion(y[:, t], x[:, t], thetas)
        pce_losses.append(pce_loss)
        nmc_losses.append(nmc_loss)
    pce_losses = torch.stack(pce_losses, dim=-1)
    nmc_losses = torch.stack(nmc_losses, dim=-1)

    # convert from EIGStepLoss output to bounds
    pce_bounds = torch.log(torch.tensor(L + 1.0)) - pce_losses
    nmc_bounds = torch.log(torch.tensor(float(L))) - nmc_losses
    return pce_bounds, nmc_bounds


@torch.no_grad()
def eval_condition(cfg, model, experiment, T, L, M, batch_size,
                   n_source: int, relevant: bool, T_src: int, logger=None):
    """Evaluate per-step sPCE/sNMC under one (n_source, relevant) condition."""
    n_iter = (M + batch_size - 1) // batch_size
    log_every = max(1, n_iter // 5)

    pce_list, nmc_list = [], []
    for it in range(n_iter):
        if logger is not None and ((it + 1) % log_every == 0 or it == 0 or it == n_iter - 1):
            logger.info(f"  minibatch {it+1}/{n_iter}")
        target_theta = experiment.sample_theta((batch_size,))
        ctx_x, ctx_y, _ = make_source_trajectories(experiment, target_theta,
                                                   n_source=n_source, T_src=T_src,
                                                   relevant=relevant)
        theta_0, xi_designs, y_outcomes = run_trace_with_context(
            model, experiment, T, batch_size, ctx_x, ctx_y, target_theta
        )
        pce_b, nmc_b = compute_stepwise_bounds(
            experiment, theta_0, xi_designs, y_outcomes, L, batch_size
        )
        pce_list.append(pce_b)
        nmc_list.append(nmc_b)

    pce = torch.cat(pce_list, dim=0)
    nmc = torch.cat(nmc_list, dim=0)
    M_eff = pce.shape[0]
    return AttrDict(
        pce_mean=pce.mean(0).cpu(), pce_se=(pce.std(0) / np.sqrt(M_eff)).cpu(),
        nmc_mean=nmc.mean(0).cpu(), nmc_se=(nmc.std(0) / np.sqrt(M_eff)).cpu(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--T", type=int, default=None, help="Adaptive trajectory length")
    parser.add_argument("--T_src", type=int, default=30, help="Length of each source trajectory")
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.model_path)))
    cfg = load_cfg(run_dir, args.config_path)

    output_path = args.output_path or os.path.join(run_dir, "eval", "ces_pce_per_step_context.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_dir = os.path.join(run_dir, "logs")
    logger = create_logger(log_dir, name="ces_pce_context")

    device = args.device or cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_device(device)

    T = args.T if args.T is not None else cfg.eval_T
    L = args.L if args.L is not None else cfg.eval_L
    M = args.M if args.M is not None else cfg.eval_M
    batch_size = args.batch_size if args.batch_size is not None else cfg.eval_batch_size
    T_src = args.T_src

    model = build_model(cfg)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    experiment = build_experiment(cfg)

    logger.info(f"Loaded model from {os.path.abspath(args.model_path)}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Output .pt path (will be written at end): {os.path.abspath(output_path)}")
    logger.info(f"Eval: T={T}, T_src={T_src}, L={L}, M={M}, batch_size={batch_size}, device={device}")

    conditions = [
        ("n1_relevant",   1, True),
        ("n5_relevant",   5, True),
        ("n1_irrelevant", 1, False),
        ("n5_irrelevant", 5, False),
    ]

    results = {}
    for name, n_source, relevant in conditions:
        logger.info(f"=== Condition: {name} (n_source={n_source}, relevant={relevant}) ===")
        bounds = eval_condition(cfg, model, experiment, T, L, M, batch_size,
                                n_source=n_source, relevant=relevant, T_src=T_src,
                                logger=logger)
        results[name] = bounds
        logger.info("Step |     sPCE (mean +/- s.e.)    |     sNMC (mean +/- s.e.)")
        logger.info("-" * 60)
        for t in range(T):
            logger.info(
                f"{t+1:>4} | {bounds.pce_mean[t].item():>9.4f} +/- {bounds.pce_se[t].item():.4f} "
                f" | {bounds.nmc_mean[t].item():>9.4f} +/- {bounds.nmc_se[t].item():.4f}"
            )

    payload = {name: {
        "pce_mean": b.pce_mean, "pce_se": b.pce_se,
        "nmc_mean": b.nmc_mean, "nmc_se": b.nmc_se,
    } for name, b in results.items()}
    payload.update({
        "T": T, "T_src": T_src, "L": L, "M": M, "batch_size": batch_size,
        "model_path": os.path.abspath(args.model_path),
    })
    torch.save(payload, output_path)
    logger.info(f"Saved per-step bounds for all conditions to: {os.path.abspath(output_path)}")
    logger.info(f"Log file directory: {os.path.abspath(log_dir)}")


if __name__ == "__main__":
    main()
