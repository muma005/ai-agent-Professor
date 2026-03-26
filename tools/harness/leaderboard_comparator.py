"""Maps a private score to a simulated leaderboard percentile and medal."""


def compare_to_leaderboard(private_score: float, spec) -> dict:
    percentile = spec.lb_curve.score_to_percentile(private_score)
    medal      = _medal(percentile)
    hib        = spec.lb_curve.higher_is_better

    gap_to_bronze = (private_score - spec.bronze_threshold) if hib else (spec.bronze_threshold - private_score)
    gap_to_gold   = (private_score - spec.gold_threshold)   if hib else (spec.gold_threshold   - private_score)

    return {
        "simulated_percentile": round(percentile, 1),
        "simulated_medal":      medal,
        "gap_to_bronze":        round(gap_to_bronze, 5),
        "gap_to_gold":          round(gap_to_gold, 5),
        "bronze_threshold":     spec.bronze_threshold,
        "silver_threshold":     spec.silver_threshold,
        "gold_threshold":       spec.gold_threshold,
    }


def _medal(percentile: float) -> str:
    if percentile >= 97:  return "Gold"
    if percentile >= 90:  return "Silver"
    if percentile >= 80:  return "Bronze"
    return "None"
