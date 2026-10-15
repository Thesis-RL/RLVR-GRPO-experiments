"""Connecting metrics on frozen groups (offline diagnostic)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr

from rlvr.rewards.aggregation import grpo_advantages


@dataclass
class GroupSignalStats:
    zero_variance_clean: bool
    advantage_sign_agreement: float | None
    advantage_cosine: float | None
    pairwise_fidelity: float | None
    spearman: float | None
    top1_preserved: float | None
    best_minus_second_clean: float
    best_minus_second_noisy: float
    near_tie_clean: bool
    near_tie_noisy: bool
    n_clean_positive: int
    mean_clean_reward: float
    mean_noisy_reward: float


def group_signal_stats(
    clean_rewards: np.ndarray,
    noisy_rewards: np.ndarray,
    *,
    near_tie_eps: float = 1e-8,
) -> GroupSignalStats:
    clean_rewards = np.asarray(clean_rewards, dtype=np.float64)
    noisy_rewards = np.asarray(noisy_rewards, dtype=np.float64)
    a_c = grpo_advantages(clean_rewards)
    a_n = grpo_advantages(noisy_rewards)

    zero_var = float(clean_rewards.std(ddof=0)) < 1e-12

    sign_agree = None
    cosine = None
    if not zero_var:
        # Exclude exact zeros from sign agreement (ties / no signal).
        mask = (np.abs(a_c) > 1e-12) & (np.abs(a_n) > 1e-12)
        if mask.any():
            sign_agree = float(np.mean(np.sign(a_c[mask]) == np.sign(a_n[mask])))
        nc = np.linalg.norm(a_c)
        nn = np.linalg.norm(a_n)
        if nc > 1e-12 and nn > 1e-12:
            cosine = float(np.dot(a_c, a_n) / (nc * nn))

    pairwise = _pairwise_fidelity(clean_rewards, noisy_rewards)
    spearman = None
    if (
        not zero_var
        and len(np.unique(clean_rewards)) > 1
        and len(np.unique(noisy_rewards)) > 1
    ):
        rho, _ = spearmanr(clean_rewards, noisy_rewards)
        spearman = float(rho) if np.isfinite(rho) else None

    top1 = None
    if not zero_var:
        clean_best = int(np.argmax(clean_rewards))
        noisy_best = set(np.flatnonzero(noisy_rewards == noisy_rewards.max()))
        top1 = float(clean_best in noisy_best)

    margins_c = _margins(clean_rewards)
    margins_n = _margins(noisy_rewards)

    return GroupSignalStats(
        zero_variance_clean=zero_var,
        advantage_sign_agreement=sign_agree,
        advantage_cosine=cosine,
        pairwise_fidelity=pairwise,
        spearman=spearman,
        top1_preserved=top1,
        best_minus_second_clean=margins_c,
        best_minus_second_noisy=margins_n,
        near_tie_clean=margins_c <= near_tie_eps,
        near_tie_noisy=margins_n <= near_tie_eps,
        n_clean_positive=int(np.sum(clean_rewards > 0)),
        mean_clean_reward=float(clean_rewards.mean()),
        mean_noisy_reward=float(noisy_rewards.mean()),
    )


def _pairwise_fidelity(clean: np.ndarray, noisy: np.ndarray) -> float | None:
    n = len(clean)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dc = clean[i] - clean[j]
            if abs(dc) < 1e-12:
                continue
            dn = noisy[i] - noisy[j]
            total += 1
            if abs(dn) < 1e-12:
                continue  # tie under noisy: count as disagreement for strict fidelity
            if np.sign(dc) == np.sign(dn):
                agree += 1
    if total == 0:
        return None
    return agree / total


def _margins(rewards: np.ndarray) -> float:
    uniq = np.sort(np.unique(rewards))[::-1]
    if len(uniq) < 2:
        return 0.0
    return float(uniq[0] - uniq[1])


def classify_signal(value: float | None, survives: float, attenuated: float) -> str:
    if value is None:
        return "undefined"
    if value >= survives:
        return "survives"
    if value >= attenuated:
        return "attenuated"
    return "destroyed"


def summarize_groups(stats_list: list[GroupSignalStats]) -> dict:
    usable = [s for s in stats_list if not s.zero_variance_clean]

    def mean_of(attr: str) -> float | None:
        vals = [getattr(s, attr) for s in usable if getattr(s, attr) is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "n_groups": len(stats_list),
        "n_zero_variance_clean": sum(1 for s in stats_list if s.zero_variance_clean),
        "n_usable": len(usable),
        "mean_advantage_sign_agreement": mean_of("advantage_sign_agreement"),
        "mean_advantage_cosine": mean_of("advantage_cosine"),
        "mean_pairwise_fidelity": mean_of("pairwise_fidelity"),
        "mean_spearman": mean_of("spearman"),
        "mean_top1_preserved": mean_of("top1_preserved"),
        "mean_clean_reward_all": float(np.mean([s.mean_clean_reward for s in stats_list]))
        if stats_list
        else None,
        "mean_noisy_reward_all": float(np.mean([s.mean_noisy_reward for s in stats_list]))
        if stats_list
        else None,
        "frac_problems_nonzero_variance": (len(usable) / len(stats_list)) if stats_list else None,
    }
    # Pre-registered buckets
    for key, survives, attenuated in (
        ("mean_advantage_sign_agreement", 0.70, 0.55),
        ("mean_advantage_cosine", 0.70, 0.40),
        ("mean_pairwise_fidelity", 0.80, 0.60),
        ("mean_top1_preserved", 0.70, 0.50),
    ):
        out[f"{key}_bucket"] = classify_signal(out[key], survives, attenuated)
    return out


def difficulty_stratum(mean_frac_reward: float) -> str:
    if mean_frac_reward >= 2 / 3:
        return "easy"
    if mean_frac_reward > 1 / 3:
        return "medium"
    return "hard"


def stats_to_dict(s: GroupSignalStats) -> dict:
    return asdict(s)
