# Adaptive Process Model Benchmark Report

**Model Identifier:** renting_log_high_textual_model
**Generated:** 2026-01-29 12:24:15

## Configuration

- **Model Identifier:** renting_log_high_textual_model
- **Textual Model File:** renting_log_high_textual_model.txt
- **XES File:** renting_log_high.xes
- **Adapter LLM:** openai:gpt-5
- **Validator LLM:** gpt-5-mini
- **Valid Traces:** 100
- **Invalid Traces:** 100
- **Tested Audiences:** executives, children
- **Baseline Mode:** Last

## Executive Summary

This benchmark evaluates how well audience-adapted process models preserve semantic fidelity for trace validation. 2 audience(s) completed successfully.

**Baseline Performance (Original Model):**
- F1 Score: 0.943
- Precision: 0.893
- Recall: 1.000

## Results by Audience

### Performance Comparison Table

| Audience | F1 Score | Δ F1 | Precision | Recall | TP | FP | TN | FN |
|----------|----------|------|-----------|--------|----|----|----|----||
| **Baseline** | 0.943 | - | 0.893 | 1.000 | 100 | 12 | 88 | 0 |
| Executives | 0.990 | +0.047 | 0.980 | 1.000 | 100 | 2 | 98 | 0 |
| Children | 0.939 | -0.004 | 0.885 | 1.000 | 100 | 13 | 87 | 0 |

### Audience Ranking

Ranked from best to worst F1 preservation:

1. **Children**: F1=0.939 (Δ -0.004)
2. **Executives**: F1=0.990 (Δ +0.047)

## Detailed Results

### Executives

**Performance:**
- F1 Score: 0.990 (+0.047 vs baseline)
- Precision: 0.980
- Recall: 1.000
- True Positives: 100
- False Positives: 2
- True Negatives: 98
- False Negatives: 0

**Resources:**
- Compute Time: 33188.6 seconds
- Total Tokens: 864,619

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 2
- Details: See `misclassified/executives/`

**Adapted Model:** `adapted_executives.txt`

---

### Children

**Performance:**
- F1 Score: 0.939 (-0.004 vs baseline)
- Precision: 0.885
- Recall: 1.000
- True Positives: 100
- False Positives: 13
- True Negatives: 87
- False Negatives: 0

**Resources:**
- Compute Time: 7608.3 seconds
- Total Tokens: 783,659

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 13
- Details: See `misclassified/children/`

**Adapted Model:** `adapted_children.txt`

---

## Observations

All 2 audience adaptation(s) completed successfully.


## Files Generated

- `config.json` - Benchmark configuration
- `original_textual_model.txt` - Baseline model
- `adapted_<audience>.txt` - 5 adapted versions
- `results.json` - Complete metrics data
- `comparison_report.md` - This report
- `traces/valid_traces.pkl` - Test valid traces
- `traces/invalid_traces.pkl` - Test invalid traces
- `misclassified/<audience>/` - Misclassified traces per audience

## Methodology

1. **Baseline Establishment:** Original textual model validated against test traces
2. **Adaptation:** Each audience receives high-quality adaptation using GPT-5
3. **Validation:** Adapted models validated using cost-effective GPT-5-mini
4. **Comparison:** F1 scores compared to measure semantic preservation
5. **Completion:** All benchmarks run to completion regardless of performance

The goal is zero information loss - adapted models should preserve all behavioral information for trace validation.
