"""Verify configs/model.yaml produces the declared parameter count.

Run this after any change to configs/model.yaml (Architecture.md §27) to
guard against a repeat of the Architecture.md §0 bug, where the documented
config silently didn't match its declared parameter count.
"""

import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
CONFIG_PATH = DEFAULT_CONFIG


def count_params(cfg: dict) -> dict:
    vocab = cfg["vocab_size"]
    d_model = cfg["d_model"]
    n_layer = cfg["n_layer"]
    d_ff = cfg["d_ff"]
    # None/absent = every depth-slot has its own weights (baseline).
    # Architecture.md §31 — loop-transformer: fewer unique blocks, cycled.
    n_unique = cfg.get("n_unique_layers") or n_layer

    embedding = vocab * d_model
    attn_per_layer = 4 * d_model * d_model  # Wq, Wk, Wv, Wo
    mlp_per_layer = 3 * d_model * d_ff  # SwiGLU: W_gate, W_up, W_down
    norm_per_layer = 2 * d_model  # pre-attn + pre-mlp RMSNorm scale
    per_layer = attn_per_layer + mlp_per_layer + norm_per_layer
    final_norm = d_model
    # Only the unique blocks contribute params; looped depth-slots reuse them.
    all_layers = n_unique * per_layer
    # Per-depth-slot conditioning vector, only present when actually looping.
    depth_embed = n_layer * d_model if n_unique < n_layer else 0

    total = embedding + all_layers + final_norm + depth_embed
    if not cfg.get("weight_tying", True):
        total += embedding  # untied LM head would add a second matrix

    return {
        "embedding": embedding,
        "per_layer": per_layer,
        "n_unique": n_unique,
        "all_layers": all_layers,
        "depth_embed": depth_embed,
        "final_norm": final_norm,
        "total": total,
    }


def main(config_path: Path = CONFIG_PATH) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    if cfg["d_head"] * cfg["n_head"] != cfg["d_model"]:
        print(
            f"FAIL: d_head ({cfg['d_head']}) * n_head ({cfg['n_head']}) "
            f"!= d_model ({cfg['d_model']})"
        )
        return 1

    breakdown = count_params(cfg)
    expected = cfg["expected_param_count"]
    actual = breakdown["total"]

    print(f"Embedding:        {breakdown['embedding']:>12,}")
    print(f"Unique layers:    {breakdown['n_unique']:>12,}  (of {cfg['n_layer']} depth-slots)")
    print(f"All layers:       {breakdown['all_layers']:>12,}")
    print(f"Depth embed:      {breakdown['depth_embed']:>12,}")
    print(f"Final norm:       {breakdown['final_norm']:>12,}")
    print(f"{'-' * 30}")
    print(f"Total:            {actual:>12,}")
    print(f"Expected:         {expected:>12,}")

    if actual != expected:
        print(f"\nFAIL: parameter count mismatch ({actual:,} != {expected:,})")
        return 1

    print(f"\nOK: parameter count matches {config_path.name} declaration.")
    return 0


if __name__ == "__main__":
    requested = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH
    sys.exit(main(requested))
