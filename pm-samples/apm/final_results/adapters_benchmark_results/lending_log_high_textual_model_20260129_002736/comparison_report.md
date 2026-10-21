# Adaptive Process Model Benchmark Report

**Model Identifier:** lending_log_high_textual_model
**Generated:** 2026-01-29 12:30:56

## Configuration

- **Model Identifier:** lending_log_high_textual_model
- **Textual Model File:** lending_log_high_textual_model.txt
- **XES File:** lending_log_high.xes
- **Adapter LLM:** openai:gpt-5
- **Validator LLM:** gpt-5-mini
- **Valid Traces:** 100
- **Invalid Traces:** 100
- **Tested Audiences:** executives, children
- **Baseline Mode:** Last

## Executive Summary

This benchmark evaluates how well audience-adapted process models preserve semantic fidelity for trace validation. 2 audience(s) completed successfully.

**Baseline Performance (Original Model):**
- F1 Score: 1.000
- Precision: 1.000
- Recall: 1.000

## Results by Audience

### Performance Comparison Table

| Audience | F1 Score | Δ F1 | Precision | Recall | TP | FP | TN | FN |
|----------|----------|------|-----------|--------|----|----|----|----||
| **Baseline** | 1.000 | - | 1.000 | 1.000 | 100 | 0 | 100 | 0 |
| Executives | 1.000 | +0.000 | 1.000 | 1.000 | 100 | 0 | 100 | 0 |
| Children | 1.000 | +0.000 | 1.000 | 1.000 | 100 | 0 | 100 | 0 |

### Audience Ranking

Ranked from best to worst F1 preservation:

1. **Executives**: F1=1.000 (Δ +0.000)
2. **Children**: F1=1.000 (Δ +0.000)

## Detailed Results

### Executives

**Performance:**
- F1 Score: 1.000 (+0.000 vs baseline)
- Precision: 1.000
- Recall: 1.000
- True Positives: 100
- False Positives: 0
- True Negatives: 100
- False Negatives: 0

**Resources:**
- Compute Time: 33503.6 seconds
- Total Tokens: 842,249

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 0
- Details: See `misclassified/executives/`

**Adapted Model:** `adapted_executives.txt`

---

### Children

**Performance:**
- F1 Score: 1.000 (+0.000 vs baseline)
- Precision: 1.000
- Recall: 1.000
- True Positives: 100
- False Positives: 0
- True Negatives: 100
- False Negatives: 0

**Resources:**
- Compute Time: 7361.6 seconds
- Total Tokens: 803,194

**Misclassified Traces:**
- False Negatives: 0
- False Positives: 0
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
