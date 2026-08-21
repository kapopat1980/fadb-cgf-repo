"""
Statistical significance testing utilities.

- mcnemar_test: paired McNemar's test on two models' predictions vs. ground truth
- fishers_method / stouffers_method: combine per-seed p-values into a single
  verdict (used because McNemar's test is inherently single-run/paired; a
  multi-seed study needs a principled way to aggregate across seeds)

See Section 3.6 and Section 4.2 of the manuscript, and the standalone
combine_mcnemar_pvalues.py script for the original analysis this was
extracted from.
"""
import numpy as np
from scipy.stats import chi2, norm
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(preds_a, preds_b, labels):
    """Returns (contingency_table, statistic, p_value) for paired predictions."""
    a_correct = np.array(preds_a) == np.array(labels)
    b_correct = np.array(preds_b) == np.array(labels)
    both = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    neither = int(np.sum(~a_correct & ~b_correct))
    table = [[both, a_only], [b_only, neither]]
    result = mcnemar(table, exact=(a_only + b_only) < 25)
    return table, float(result.statistic), float(result.pvalue)


def fishers_method(p_values):
    """Combines independent p-values via Fisher's method:
    -2*sum(ln(p_i)) ~ chi2 with 2k degrees of freedom.
    """
    p_values = np.array([p for p in p_values if p is not None and 0 < p <= 1])
    if len(p_values) == 0:
        return None, None
    statistic = -2.0 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    combined_p = 1.0 - chi2.cdf(statistic, df)
    return float(statistic), float(combined_p)


def stouffers_method(p_values):
    """Alternative combination method (Stouffer's Z-score), used as a
    cross-check since Fisher's method can be oversensitive to one small p."""
    p_values = np.array([p for p in p_values if p is not None and 0 < p < 1])
    if len(p_values) == 0:
        return None, None
    z_scores = norm.isf(p_values)
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    combined_p = norm.sf(combined_z)
    return float(combined_z), float(combined_p)
