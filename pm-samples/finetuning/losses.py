"""Reward/loss functions for REINFORCE-based fine-tuning.

Each function takes (TP, FP, TN, FN) from trace classification and returns a
scalar reward in [-1, 1] or [0, 1]. Higher = better textual model.

Context from the thesis evaluation (Ch.5):
- Most configurations exhibit near-perfect recall (FN ~0) but variable precision
- The dominant error pattern is "permissive" models: high FP, low FN
- Refinement that pushes FP too low can cause "semantic overfitting" (sudden FN spike)
- The sweet spot is low FP while maintaining low FN
- Hiring is hardest (most FP), Hospital is easiest
"""

from typing import Callable, Dict
import math


# ---------------------------------------------------------------------------
# Standard metrics as rewards
# ---------------------------------------------------------------------------

def f1_reward(TP: int, FP: int, TN: int, FN: int) -> float:
    """Standard F1 score. Balanced precision-recall harmonic mean.
    Baseline reward function. Treats FP and FN symmetrically."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def fbeta_precision_reward(TP: int, FP: int, TN: int, FN: int, beta: float = 0.5) -> float:
    """F-beta score with beta < 1 to emphasize precision over recall.
    beta=0.5 weights precision 2x more than recall.
    Rationale: since models tend to have perfect recall but poor precision,
    this reward pushes harder on reducing FP without ignoring FN."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def mcc_reward(TP: int, FP: int, TN: int, FN: int) -> float:
    """Matthews Correlation Coefficient, rescaled to [0, 1].
    MCC ranges from -1 to 1; we map it to [0, 1] for use as a reward.
    MCC is considered one of the most balanced metrics for binary classification,
    accounting for all four confusion matrix cells."""
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denom == 0:
        return 0.0
    mcc = (TP * TN - FP * FN) / denom
    return (mcc + 1.0) / 2.0  # rescale [-1, 1] -> [0, 1]


def balanced_accuracy_reward(TP: int, FP: int, TN: int, FN: int) -> float:
    """Balanced accuracy: average of sensitivity and specificity.
    Sensitivity (recall) = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Less sensitive to class imbalance than standard accuracy."""
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


# ---------------------------------------------------------------------------
# Custom rewards targeting the observed FP-dominant error pattern
# ---------------------------------------------------------------------------

def asymmetric_fp_penalty(TP: int, FP: int, TN: int, FN: int, fp_weight: float = 2.0, fn_weight: float = 1.0) -> float:
    """Asymmetric penalty that weighs false positives more heavily.
    Rationale: textual models are consistently too permissive (high FP, low FN).
    By penalizing FP more, we push the model to generate more precise descriptions
    that the validator can use to correctly reject invalid traces.

    reward = 1 - (fp_weight * FP + fn_weight * FN) / (fp_weight * FP + fn_weight * FN + TP + TN)
    When fp_weight > fn_weight, reducing FP yields more reward than reducing FN."""
    total = TP + FP + TN + FN
    if total == 0:
        return 0.0
    weighted_errors = fp_weight * FP + fn_weight * FN
    weighted_total = weighted_errors + TP + TN
    return 1.0 - weighted_errors / weighted_total


def precision_at_high_recall(TP: int, FP: int, TN: int, FN: int, recall_floor: float = 0.95) -> float:
    """Maximize precision while maintaining recall above a threshold.
    Rationale: the thesis shows that most models achieve recall=1.0.
    We want to keep that property while pushing precision as high as possible.
    If recall drops below recall_floor, apply a harsh penalty.

    Returns:
        precision if recall >= recall_floor
        precision * (recall / recall_floor)^4 otherwise (steep penalty)"""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if recall >= recall_floor:
        return precision
    penalty = (recall / recall_floor) ** 4
    return precision * penalty


def anti_permissiveness(TP: int, FP: int, TN: int, FN: int) -> float:
    """Reward that specifically targets the "permissive model" problem.
    Combines specificity (ability to reject invalid traces) with a recall gate.

    specificity = TN / (TN + FP)  -- how well the model rejects invalid traces
    recall_gate = 1 if recall >= 0.9, else steep decay

    This directly addresses the thesis finding that models accept ALL traces
    (both valid and invalid) because the textual description is too loose."""
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if recall >= 0.9:
        recall_gate = 1.0
    else:
        recall_gate = (recall / 0.9) ** 3
    return specificity * recall_gate


def adaptive_fp_fn_balance(TP: int, FP: int, TN: int, FN: int) -> float:
    """Adaptive reward that shifts focus based on current error distribution.
    When FP >> FN (permissive): emphasizes precision improvement.
    When FN >> FP (over-specified): emphasizes recall improvement.
    When FP ~ FN: falls back to standard F1.

    This follows the thesis observation that refinement oscillates between
    permissive (high FP) and over-specified (high FN) states."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    total_errors = FP + FN
    if total_errors == 0:
        return 1.0
    fp_ratio = FP / total_errors
    # fp_ratio near 1.0 => mostly FP errors => weight precision more (beta < 1)
    # fp_ratio near 0.0 => mostly FN errors => weight recall more (beta > 1)
    beta = 1.0 / (1.0 + fp_ratio)  # ranges (0.5, 1.0) roughly
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


# ---------------------------------------------------------------------------
# Suggestions for additional loss functions to explore
# ---------------------------------------------------------------------------
#
# 1. Tversky Loss (as reward):
#    Generalization of F1 where FP and FN have independent weights alpha and beta.
#    reward = TP / (TP + alpha*FP + beta*FN)
#    With alpha > beta, directly penalizes FP more. Widely used in medical image
#    segmentation for the same class-imbalance pattern.
#
# 2. Soft Specificity-Sensitivity product:
#    reward = sensitivity^a * specificity^b with tunable exponents.
#    When b > a, pushes specificity (invalid trace rejection) harder.
#
# 3. Log-barrier recall reward:
#    reward = precision - lambda * max(0, log(recall_floor / recall))
#    Keeps recall above floor via log-barrier, optimizes precision freely above it.
#    Inspired by interior-point methods in constrained optimization.
#
# 4. Curriculum-aware reward:
#    Start with f1_reward (balanced), then anneal toward asymmetric_fp_penalty
#    as training progresses. Early epochs learn general structure, later epochs
#    focus on precision improvement.
#
# 5. Per-process adaptive reward:
#    Maintain per-process running stats of FP/FN ratio, and select the most
#    appropriate reward function per problem. Hiring (high FP) gets
#    anti_permissiveness, Hospital (near-perfect) gets f1_reward.
#
# 6. Contrastive trace-level reward:
#    Instead of aggregate metrics, assign per-trace rewards:
#    +1 for correct classification, -1 for misclassification (weighted by type).
#    Requires modifying the REINFORCE objective to sum per-trace rewards.
#


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY: Dict[str, Callable] = {
    "f1": f1_reward,
    "fbeta_precision": fbeta_precision_reward,
    "mcc": mcc_reward,
    "balanced_accuracy": balanced_accuracy_reward,
    "asymmetric_fp": asymmetric_fp_penalty,
    "precision_at_high_recall": precision_at_high_recall,
    "anti_permissiveness": anti_permissiveness,
    "adaptive_fp_fn": adaptive_fp_fn_balance,
}


def get_loss_fn(name: str) -> Callable:
    """Get a reward function by name."""
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss function '{name}'. Available: {available}")
    return LOSS_REGISTRY[name]


def list_loss_functions() -> Dict[str, str]:
    """List available loss functions with their docstrings."""
    return {name: fn.__doc__.split("\n")[0] for name, fn in LOSS_REGISTRY.items()}
