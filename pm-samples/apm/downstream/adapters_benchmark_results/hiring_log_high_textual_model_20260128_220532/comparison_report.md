# Adaptive Process Model Benchmark Report

**Model Identifier:** hiring_log_high_textual_model
**Generated:** 2026-01-29 00:12:36

## Configuration

- **Model Identifier:** hiring_log_high_textual_model
- **Textual Model File:** hiring_log_high_textual_model.txt
- **XES File:** hiring_log_high.xes
- **Adapter LLM:** openai:gpt-5
- **Validator LLM:** gpt-5-mini
- **Valid Traces:** 100
- **Invalid Traces:** 100
- **Tested Audiences:** executives, children
- **Baseline Mode:** Last

## Executive Summary

This benchmark evaluates how well audience-adapted process models preserve semantic fidelity for trace validation. 2 audience(s) completed successfully.

**Baseline Performance (Original Model):**
- F1 Score: 0.873
- Precision: 0.775
- Recall: 1.000

## Results by Audience

### Performance Comparison Table

| Audience | F1 Score | Δ F1 | Precision | Recall | TP | FP | TN | FN |
|----------|----------|------|-----------|--------|----|----|----|----||
| **Baseline** | 0.873 | - | 0.775 | 1.000 | 100 | 29 | 71 | 0 |
| Executives | 0.889 | +0.016 | 0.800 | 1.000 | 100 | 25 | 75 | 0 |
| Children | 0.943 | +0.070 | 0.893 | 1.000 | 100 | 12 | 88 | 0 |

### Audience Ranking

Ranked from best to worst F1 preservation:

1. **Executives**: F1=0.889 (Δ +0.016)
2. **Children**: F1=0.943 (Δ +0.070)

## Detailed Results

### Executives

**Performance:**
- F1 Score: 0.889 (+0.016 vs baseline)
- Precision: 0.800
- Recall: 1.000
- True Positives: 100
- False Positives: 25
- True Negatives: 75
- False Negatives: 0

**Resources:**
- Compute Time: 2199.5 seconds
- Total Tokens: 933,074

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 25
- Details: See `misclassified/executives/`

**Adapted Model:** `adapted_executives.txt`

---

### Children

**Performance:**
- F1 Score: 0.943 (+0.070 vs baseline)
- Precision: 0.893
- Recall: 1.000
- True Positives: 100
- False Positives: 12
- True Negatives: 88
- False Negatives: 0

**Resources:**
- Compute Time: 2381.0 seconds
- Total Tokens: 886,610

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 12
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
