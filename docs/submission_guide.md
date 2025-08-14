# Submission Guide

## How to Submit Results

### 1. Run Evaluation
```bash
qeval run q-humaneval your-model --num-samples 50 
```

### 2. Prepare Files
- **Results file**: `results_your-model_timestamp.json` (auto-generated)
- **Model card**: Create `model_cards/your-model.md`

### 3. Submit Pull Request
1. Fork the repository
2. Add both files to your fork
3. Submit PR with title: `Add results: your-model`

## Model Card Template

```markdown
# Model Name

**Provider**: Organization/Individual
**Version**: v1.0
**Parameters**: 7B/13B/etc
**License**: MIT/Apache/etc

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: YYYY-MM-DD
- **Hardware**: GPU/CPU specs

## Results Summary
- **Pass@1**: X.X%
- **Pass@5**: X.X%
- **Pass@10**: X.X%
```

## Requirements
- Minimum 50 samples per problem for statistical significance
- Complete model card with all required fields
- Valid JSON results file format
- One model per PR

### Statistical Rigor: Why 50 Samples Matter
For reliable Pass@k (k <= 10) evaluation, the sample size is critical. Using Wilson confidence intervals with independent seeds, we determined that Q-HumanEval (164 problems) requires at least 50 samples per problem to achieve statistically significant results with ±3 percentage point confidence intervals at 95% confidence level. This conservative estimate accounts for worst-case variance and ensures meaningful model comparisons.

## Review Process
PRs are reviewed for:
- Results file format and completeness
- Model card accuracy
- Statistical validity
- Proper attribution
