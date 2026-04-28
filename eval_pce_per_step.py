""" Evaluate per-step sPCE / sNMC bounds for the Location Finding task.

Defaults assume the trained checkpoint lives at ``./models/loc.pth`` (drop the
``.pth`` from training there). Outputs go to ``./out/eval/``:
    - loc_pce_per_step.pt   easy-to-read flat dict (see ``payload`` below)
    - loc_pce_per_step.txt  human-readable per-step table

The model was trained with T=20 but per-step bounds are evaluated up to T=30
by default to inspect extrapolation behaviour.
"""
import argparse
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data import HiddenLocation
from model.mlp import EncoderNetwork, EmitterNetwork, SetEquivariantDesignNetwork
from utils import create_logger, set_seed
from utils.eval import eval_bounds


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "loc.pth")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "out", "eval")


def load_cfg(config_path: str = None):
    """Compose the location-finding config from ``./config`` defaults."""
    if config_path is not None and os.path.isfile(config_path):
        return OmegaConf.load(config_path)
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(REPO_ROOT, "config", "loc.yaml")),
        {"data": OmegaConf.load(os.path.join(REPO_ROOT, "config", "data", "location.yaml"))},
        {"model": OmegaConf.load(os.path.join(REPO_ROOT, "config", "model", "mlp_loc.yaml"))},
    )
    return cfg


def build_model(cfg):
    encoder = EncoderNetwork(
        cfg.data.dim_design, cfg.data.dim_outcome,
        hidden_dim=cfg.model.hidden_dim, encoding_dim=cfg.model.encoding_dim,
        hidden_depth=cfg.model.encoder_hidden_depth, activation=nn.ReLU(),
    )
    emitter = EmitterNetwork(
        cfg.model.encoding_dim, cfg.data.dim_design,
        hidden_dim=cfg.model.hidden_dim, hidden_depth=cfg.model.emitter_hidden_depth,
        activation=nn.Identity(),
    )
    model = SetEquivariantDesignNetwork(
        encoder, emitter,
        cfg.data.dim_design, cfg.data.dim_outcome,
        empty_value=torch.ones(cfg.model.encoding_dim) * 0.01,
    )
    return model


def build_experiment(cfg):
    return HiddenLocation(
        dim=cfg.data.dim_design, K=cfg.data.K, theta_dist=cfg.data.theta_dist,
        design_scale=cfg.data.design_scale, noise_scale=cfg.data.noise_scale,
        base_signal=cfg.data.base_signal, max_signal=cfg.data.max_signal,
    )


def write_table(path, header_lines, T, pce_mean, pce_se, nmc_mean, nmc_se):
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(f"{'step':>4}  {'pce_mean':>10}  {'pce_se':>10}  {'nmc_mean':>10}  {'nmc_se':>10}\n")
        for t in range(T):
            f.write(
                f"{t+1:>4}  {pce_mean[t].item():>10.4f}  {pce_se[t].item():>10.4f}  "
                f"{nmc_mean[t].item():>10.4f}  {nmc_se[t].item():>10.4f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Trained model state dict (default: ./models/loc.pth)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config; default composes ./config/loc.yaml")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write loc_pce_per_step.{pt,txt}")
    parser.add_argument("--T", type=int, default=30, help="Per-step eval horizon (default 30)")
    parser.add_argument("--L", type=int, default=None, help="Override eval_L from config")
    parser.add_argument("--M", type=int, default=None, help="Override eval_M from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval_batch_size")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    cfg = load_cfg(args.config_path)

    device = args.device or cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_device(device)

    set_seed(args.seed)

    T = args.T
    L = args.L if args.L is not None else cfg.eval_L
    M = args.M if args.M is not None else cfg.eval_M
    batch_size = args.batch_size if args.batch_size is not None else cfg.eval_batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    logger = create_logger(log_dir, name="loc_pce")

    model = build_model(cfg)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    experiment = build_experiment(cfg)

    logger.info(f"Loaded model from {os.path.abspath(args.model_path)}")
    logger.info(f"Eval: T={T}, L={L}, M={M}, batch_size={batch_size}, device={device}, seed={args.seed}")

    bounds = eval_bounds(cfg, model, experiment, T=T, L=L, M=M, batch_size=batch_size, stepwise=True)

    logger.info("Step |     sPCE (mean +/- s.e.)    |     sNMC (mean +/- s.e.)")
    logger.info("-" * 60)
    for t in range(T):
        logger.info(
            f"{t+1:>4} | {bounds.pce_mean[t].item():>9.4f} +/- {bounds.pce_se[t].item():.4f} "
            f" | {bounds.nmc_mean[t].item():>9.4f} +/- {bounds.nmc_se[t].item():.4f}"
        )

    pt_path = os.path.join(args.output_dir, "loc_pce_per_step.pt")
    txt_path = os.path.join(args.output_dir, "loc_pce_per_step.txt")

    payload = {
        "task": "location_finding",
        "T": T, "L": L, "M": M, "batch_size": batch_size,
        "seed": args.seed,
        "model_path": os.path.abspath(args.model_path),
        "device": device,
        "step": torch.arange(1, T + 1),
        "pce_mean": bounds.pce_mean.cpu(),
        "pce_se": bounds.pce_se.cpu(),
        "nmc_mean": bounds.nmc_mean.cpu(),
        "nmc_se": bounds.nmc_se.cpu(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, pt_path)

    header_lines = [
        "Location finding per-step sPCE / sNMC bounds",
        f"T={T}  L={L}  M={M}  batch_size={batch_size}  seed={args.seed}",
        f"model: {os.path.abspath(args.model_path)}",
    ]
    write_table(txt_path, header_lines, T,
                bounds.pce_mean, bounds.pce_se, bounds.nmc_mean, bounds.nmc_se)

    logger.info(f"Saved per-step bounds to {pt_path}")
    logger.info(f"Saved per-step table  to {txt_path}")


if __name__ == "__main__":
    main()
