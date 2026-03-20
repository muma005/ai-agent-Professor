# tools/stability_validator.py

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_SEEDS      = [42, 7, 123, 999, 2024]
STABILITY_PENALTY  = 1.5   # multiplier on std in stability_score formula


@dataclass
class StabilityResult:
    mean:             float
    std:              float
    stability_score:  float      # mean - 1.5 * std
    seed_results:     list[float]
    seeds_used:       list[int]
    min_score:        float
    max_score:        float
    spread:           float      # max - min


def run_with_seeds(
    config: dict,
    train_fn,                    # callable(config, seed) -> float (cv score)
    seeds: list[int] = None,
    penalty: float = STABILITY_PENALTY,
) -> StabilityResult:
    """
    Runs a training configuration with multiple random seeds.
    Returns mean, std, and stability_score = mean - penalty * std.

    Never raises — returns result with available seeds if some fail.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    seed_results = []
    seeds_used   = []

    for seed in seeds:
        try:
            score = float(train_fn(config, seed))
            seed_results.append(score)
            seeds_used.append(seed)
            logger.debug(
                f"[stability_validator] seed={seed}: score={score:.6f}"
            )
        except Exception as e:
            logger.warning(
                f"[stability_validator] seed={seed} failed: {e}. Skipping."
            )

    if not seed_results:
        logger.warning(
            "[stability_validator] All seeds failed. Returning zero stability score."
        )
        return StabilityResult(
            mean=0.0, std=0.0, stability_score=0.0,
            seed_results=[], seeds_used=[],
            min_score=0.0, max_score=0.0, spread=0.0,
        )

    mean  = float(np.mean(seed_results))
    std   = float(np.std(seed_results))
    stab  = mean - penalty * std

    result = StabilityResult(
        mean=round(mean, 6),
        std=round(std, 6),
        stability_score=round(stab, 6),
        seed_results=[round(s, 6) for s in seed_results],
        seeds_used=seeds_used,
        min_score=round(float(min(seed_results)), 6),
        max_score=round(float(max(seed_results)), 6),
        spread=round(float(max(seed_results) - min(seed_results)), 6),
    )

    logger.info(
        f"[stability_validator] "
        f"mean={result.mean:.5f}, std={result.std:.5f}, "
        f"stability_score={result.stability_score:.5f}, "
        f"spread={result.spread:.5f} "
        f"({len(seeds_used)}/{len(seeds)} seeds succeeded)"
    )

    return result


def rank_by_stability(
    configs: list[dict],
    stability_results: list[StabilityResult],
) -> list[tuple[dict, StabilityResult]]:
    """
    Ranks (config, result) pairs by stability_score descending.
    Most stable config first.
    """
    if len(configs) != len(stability_results):
        raise ValueError(
            f"len(configs)={len(configs)} != len(stability_results)={len(stability_results)}"
        )

    paired = list(zip(configs, stability_results))
    paired.sort(key=lambda x: x[1].stability_score, reverse=True)
    return paired


def format_stability_report(
    ranked: list[tuple[dict, StabilityResult]],
    top_n: int = 5,
) -> str:
    """
    Formats a human-readable stability ranking report for lineage logging.
    """
    lines = [f"Top {min(top_n, len(ranked))} configs by stability score:"]
    for i, (config, result) in enumerate(ranked[:top_n]):
        lines.append(
            f"  [{i+1}] stability={result.stability_score:.5f} "
            f"(mean={result.mean:.5f}, std={result.std:.5f}, "
            f"spread={result.spread:.5f})"
        )
    return "\n".join(lines)
