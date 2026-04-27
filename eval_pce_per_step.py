""" Evaluate sPCE bounds at every step of the location finding task and save to .pt """
import argparse
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data import HiddenLocation
from model.mlp import EncoderNetwork, EmitterNetwork, SetEquivariantDesignNetwork
from utils.eval import eval_bounds


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model state dict, e.g. outputs/.../model/loc.pth")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to .hydra/config.yaml. Default: <run_dir>/.hydra/config.yaml")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save the per-step bounds .pt file. Default: <run_dir>/eval/loc_pce_per_step.pt")
    parser.add_argument("--T", type=int, default=None, help="Override eval_T")
    parser.add_argument("--L", type=int, default=None, help="Override eval_L")
    parser.add_argument("--M", type=int, default=None, help="Override eval_M")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval_batch_size")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda")
    args = parser.parse_args()

    # Resolve run dir = parent of "model/" directory containing the checkpoint
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.model_path)))
    config_path = args.config_path or os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Pass --config_path explicitly.")

    cfg = OmegaConf.load(config_path)

    # Device
    device = args.device or cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_device(device)

    # Eval overrides
    T = args.T if args.T is not None else cfg.eval_T
    L = args.L if args.L is not None else cfg.eval_L
    M = args.M if args.M is not None else cfg.eval_M
    batch_size = args.batch_size if args.batch_size is not None else cfg.eval_batch_size

    # Build model and experiment
    model = build_model(cfg)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    experiment = build_experiment(cfg)

    print(f"Loaded model from {args.model_path}")
    print(f"Running stepwise evaluation: T={T}, L={L}, M={M}, batch_size={batch_size}, device={device}")

    bounds = eval_bounds(cfg, model, experiment, T=T, L=L, M=M, batch_size=batch_size, stepwise=True)

    # Print per-step PCE
    print("\nStep |     sPCE (mean ± s.e.)    |     sNMC (mean ± s.e.)")
    print("-" * 60)
    for t in range(T):
        print(f"{t+1:>4} | {bounds.pce_mean[t].item():>9.4f} ± {bounds.pce_se[t].item():.4f} "
              f" | {bounds.nmc_mean[t].item():>9.4f} ± {bounds.nmc_se[t].item():.4f}")

    # Save
    output_path = args.output_path or os.path.join(run_dir, "eval", "loc_pce_per_step.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "pce_mean": bounds.pce_mean,
        "pce_se": bounds.pce_se,
        "nmc_mean": bounds.nmc_mean,
        "nmc_se": bounds.nmc_se,
        "T": T, "L": L, "M": M, "batch_size": batch_size,
        "model_path": os.path.abspath(args.model_path),
    }
    torch.save(payload, output_path)
    print(f"\nSaved per-step bounds to {output_path}")


if __name__ == "__main__":
    main()
