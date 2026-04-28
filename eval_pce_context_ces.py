""" Evaluate per-step sPCE / sNMC bounds for the CES task across five context conditions.

Conditions:
    1. baseline       : original DAD, no context injected
    2. n1_relevant    : 1 relevant source trajectory in context
    3. n5_relevant    : 5 relevant source trajectories in context
    4. n1_irrelevant  : 1 irrelevant source trajectory in context
    5. n5_irrelevant  : 5 irrelevant source trajectories in context

Relevance follows the perturbation scheme below:
    - relevant   : source theta = target theta + N(0, (0.1*range)^2)
    - irrelevant : source theta = target theta shifted 0.3..0.5 * range away from
                   the midpoint of each component's range

The model is trained with T=10 but per-step bounds are evaluated up to T=20 by
default to inspect extrapolation. Each condition is evaluated under an identical
seed so that the target thetas (and the trace ordering) are aligned across
conditions, making the only source of variation the context set.

Outputs (default ./out/eval/):
    - ces_pce_context.pt   easy-to-read flat dict
    - ces_pce_context.txt  per-condition per-step table
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
from utils import create_logger, set_seed


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "ces.pth")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "out", "eval")

# Per-component theta ranges used to define relevant / irrelevant perturbations.
# rho in [0.01, 1.0]; alpha components in [0, 1] (re-projected to simplex);
# log_u uses ~99% of the N(1, 3) prior support.
THETA_RANGE_MIN = torch.tensor([0.01, 0.0, 0.0, 0.0, -8.0])
THETA_RANGE_MAX = torch.tensor([1.0,  1.0, 1.0, 1.0, 10.0])


def load_cfg(config_path: str = None):
    """Compose the CES config from ``./config`` defaults."""
    if config_path is not None and os.path.isfile(config_path):
        return OmegaConf.load(config_path)
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(REPO_ROOT, "config", "ces.yaml")),
        {"data": OmegaConf.load(os.path.join(REPO_ROOT, "config", "data", "ces.yaml"))},
        {"model": OmegaConf.load(os.path.join(REPO_ROOT, "config", "model", "mlp_ces.yaml"))},
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
    target_exp = target_theta.unsqueeze(1).expand(-1, n_source, -1)

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
    source[..., 0] = torch.clamp(source[..., 0], 0.01, 1.0)
    source[..., 1:4] = torch.abs(source[..., 1:4]) + 1e-6
    source[..., 1:4] = source[..., 1:4] / source[..., 1:4].sum(dim=-1, keepdim=True)
    return source


@torch.no_grad()
def make_source_trajectories(experiment: CES, target_theta: torch.Tensor,
                             n_source: int, T_src: int, relevant: bool):
    """Generate source trajectories: random baskets + simulated CES outcomes.

    Returns:
        source_x_flat [B, n_source * T_src, 6]
        source_y_flat [B, n_source * T_src, 1]
        source_theta  [B, n_source, 5]
    """
    B = target_theta.shape[0]
    D = target_theta.shape[1]
    device = target_theta.device

    source_theta = make_source_thetas(target_theta, n_source, relevant)

    basket_dim = experiment.basket_dim
    basket1 = torch.rand(B, n_source, T_src, basket_dim, device=device) * experiment.design_scale
    basket2 = torch.rand(B, n_source, T_src, basket_dim, device=device) * experiment.design_scale
    source_x = torch.cat([basket1, basket2], dim=-1)

    flat_x = source_x.reshape(-1, experiment.dim_design)
    flat_theta = source_theta.unsqueeze(2).expand(-1, -1, T_src, -1).reshape(-1, D)
    flat_y = experiment.forward(flat_x, flat_theta)
    source_y = flat_y.reshape(B, n_source, T_src, experiment.dim_outcome)

    source_x_flat = source_x.reshape(B, n_source * T_src, experiment.dim_design)
    source_y_flat = source_y.reshape(B, n_source * T_src, experiment.dim_outcome)
    return source_x_flat, source_y_flat, source_theta


@torch.no_grad()
def run_trace_with_context(model, experiment, T: int, batch_size: int,
                           context_x: torch.Tensor, context_y: torch.Tensor,
                           target_theta: torch.Tensor):
    """Run T-step adaptive trajectories, prepending context_x/y to encoder history."""
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
    """Per-step cumulative sPCE / sNMC bounds for a set of completed trajectories."""
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

    pce_bounds = torch.log(torch.tensor(L + 1.0)) - pce_losses
    nmc_bounds = torch.log(torch.tensor(float(L))) - nmc_losses
    return pce_bounds, nmc_bounds


@torch.no_grad()
def eval_condition(model, experiment, T, L, M, batch_size,
                   n_source: int, relevant: bool, T_src: int, logger=None):
    """Evaluate per-step sPCE/sNMC under one (n_source, relevant) condition.

    n_source=0 means the baseline (no context injected).
    """
    n_iter = (M + batch_size - 1) // batch_size
    log_every = max(1, n_iter // 5)

    pce_list, nmc_list = [], []
    for it in range(n_iter):
        if logger is not None and ((it + 1) % log_every == 0 or it == 0 or it == n_iter - 1):
            logger.info(f"  minibatch {it+1}/{n_iter}")
        target_theta = experiment.sample_theta((batch_size,))

        if n_source == 0:
            ctx_x = torch.empty((batch_size, 0, experiment.dim_design),
                                device=target_theta.device)
            ctx_y = torch.empty((batch_size, 0, experiment.dim_outcome),
                                device=target_theta.device)
        else:
            ctx_x, ctx_y, _ = make_source_trajectories(
                experiment, target_theta, n_source=n_source,
                T_src=T_src, relevant=relevant,
            )

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


def write_table(path, header_lines, T, results_by_name):
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        for name, b in results_by_name.items():
            f.write(f"\n## condition: {name}\n")
            f.write(f"{'step':>4}  {'pce_mean':>10}  {'pce_se':>10}  {'nmc_mean':>10}  {'nmc_se':>10}\n")
            for t in range(T):
                f.write(
                    f"{t+1:>4}  {b.pce_mean[t].item():>10.4f}  {b.pce_se[t].item():>10.4f}  "
                    f"{b.nmc_mean[t].item():>10.4f}  {b.nmc_se[t].item():>10.4f}\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Trained model state dict (default: ./models/ces.pth)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config; default composes ./config/ces.yaml")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write ces_pce_context.{pt,txt}")
    parser.add_argument("--T", type=int, default=20, help="Per-step eval horizon (default 20)")
    parser.add_argument("--T_src", type=int, default=10, help="Length of each source trajectory")
    parser.add_argument("--L", type=int, default=None, help="Override eval_L from config")
    parser.add_argument("--M", type=int, default=None, help="Override eval_M from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval_batch_size")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (re-applied per condition)")
    args = parser.parse_args()

    cfg = load_cfg(args.config_path)

    device = args.device or cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_device(device)

    T = args.T
    T_src = args.T_src
    L = args.L if args.L is not None else cfg.eval_L
    M = args.M if args.M is not None else cfg.eval_M
    batch_size = args.batch_size if args.batch_size is not None else cfg.eval_batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    logger = create_logger(log_dir, name="ces_pce_context")

    model = build_model(cfg)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    experiment = build_experiment(cfg)

    logger.info(f"Loaded model from {os.path.abspath(args.model_path)}")
    logger.info(f"Eval: T={T}, T_src={T_src}, L={L}, M={M}, batch_size={batch_size}, "
                f"device={device}, seed={args.seed}")

    # (label, n_source, relevant). n_source=0 marks the baseline (no context).
    conditions = [
        ("baseline",      0, False),
        ("n1_relevant",   1, True),
        ("n5_relevant",   5, True),
        ("n1_irrelevant", 1, False),
        ("n5_irrelevant", 5, False),
    ]

    results = {}
    for name, n_source, relevant in conditions:
        # Re-seed before each condition so target thetas / trace order line up.
        set_seed(args.seed)
        logger.info(f"=== Condition: {name} (n_source={n_source}, relevant={relevant}) ===")
        bounds = eval_condition(model, experiment, T, L, M, batch_size,
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

    pt_path = os.path.join(args.output_dir, "ces_pce_context.pt")
    txt_path = os.path.join(args.output_dir, "ces_pce_context.txt")

    payload = {
        "task": "ces",
        "T": T, "T_src": T_src, "L": L, "M": M, "batch_size": batch_size,
        "seed": args.seed,
        "model_path": os.path.abspath(args.model_path),
        "device": device,
        "step": torch.arange(1, T + 1),
        "conditions": [name for name, _, _ in conditions],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    for name, b in results.items():
        payload[name] = {
            "pce_mean": b.pce_mean, "pce_std": b.pce_se,
            "nmc_mean": b.nmc_mean, "nmc_se": b.nmc_se,
        }
    torch.save(payload, pt_path)

    header_lines = [
        "CES per-step sPCE / sNMC bounds across context conditions",
        f"T={T}  T_src={T_src}  L={L}  M={M}  batch_size={batch_size}  seed={args.seed}",
        f"model: {os.path.abspath(args.model_path)}",
        "conditions: baseline, n1_relevant, n5_relevant, n1_irrelevant, n5_irrelevant",
    ]
    write_table(txt_path, header_lines, T, results)

    logger.info(f"Saved per-step bounds to {pt_path}")
    logger.info(f"Saved per-step table  to {txt_path}")


if __name__ == "__main__":
    main()
