# Group 65 – File and Task Assignment

This document shows which member should add which files and what evidence each person should create.

## Muhammad Kumail Abbas – S397291
**Branch:** `muhammad-data-eda`

### Files to add or review
- `src/08_objective2_eda.py`
- `documentation/meeting_week1.md` (add real meeting notes)
- `documentation/contribution_log.md` (add own contribution)
- Any EDA/correlation output files produced by the final script

### Main responsibilities
- Source/data validation
- Regression 2.1 EDA
- Correlation checks
- Independent review of Regression 2.2 methodology

### Suggested real commit messages
- `Validate Objective 2 source data and row counts`
- `Review Regression 2.1 EDA and correlation results`
- `Document predictor correlation findings`
- `Review Regression 2.2 methodology`

### Pull Request title
`Complete Objective 2.1 data validation and EDA review`

---

## Roman Bisural – S397541
**Branch:** `roman-data-regression21`

### Files to add
- `src/05_objective2_prepare_datasets_CORRECTED.py`
- `src/06_objective2_linear_regression_2_1.py`
- `src/09_objective2_model_comparison.py`
- Relevant prepared datasets / split manifests / model-comparison outputs
- `documentation/meeting_week1.md`
- `documentation/meeting_week2.md`

### Main responsibilities
- Dataset preparation
- Leakage safeguards
- Regression 2.1 modelling
- Chronological train/test split
- Expanding CV
- Model comparison
- Independent review of Regression 2.2 robustness/reproducibility

### Suggested real commit messages
- `Prepare final Objective 2 regression datasets`
- `Add same-date leakage safeguards`
- `Implement date-grouped chronological split`
- `Add expanding CV for Regression 2.1`
- `Add model comparison and CV stability outputs`

### Pull Request title
`Complete Regression 2.1 modelling and leakage-safe validation`

---

## Suriya Sankar – S397448
**Branch:** `suriya-regression22`

### Files to add or review
- `src/07_objective2_linear_regression_2_2.py`
- `src/08_objective2_eda.py` (if jointly reviewed, do not overwrite another person's changes without merging)
- Relevant Regression 2.2 outputs
- `documentation/meeting_week2.md`

### Main responsibilities
- Regression 2.2 modelling
- 208-row team-match validation
- Regression 2.2 EDA
- Predictor interpretation
- Independent review of Regression 2.1 CV stability and limitations

### Suggested real commit messages
- `Implement Regression 2.2 model evaluation`
- `Validate paired team-match splitting`
- `Add Regression 2.2 EDA interpretation`
- `Review Regression 2.1 model stability`

### Pull Request title
`Complete Regression 2.2 modelling and EDA`

---

## Khushi – S398777
**Branch:** `khushi-robustness-qa`

### Files to add
- `src/10_objective2_hd_diagnostics.py`
- `README.md` (final review/update)
- `documentation/final_qa_checklist.md`
- `documentation/meeting_week3.md`
- Relevant robustness/figure outputs

### Main responsibilities
- Training-only robustness/sensitivity analysis
- Figure and visualisation review
- Cross-task synthesis
- Final report/code QA
- README final check

### Suggested real commit messages
- `Add training-only regularisation sensitivity analysis`
- `Document Objective 2 robustness findings`
- `Add final report reproducibility checklist`
- `Complete final Objective 2 QA review`

### Pull Request title
`Add robustness diagnostics and final QA documentation`

---

# Cross-Review Plan
Each Pull Request should be reviewed by another member.

| PR Author | Reviewer |
|---|---|
| Muhammad | Suriya |
| Roman | Khushi |
| Suriya | Muhammad |
| Khushi | Roman |

A useful review comment should state what was actually checked, for example:
- row counts
- predictor count
- leakage safeguards
- model-selection logic
- final metrics
- figure/report agreement

Do not leave only generic comments such as “looks good”.
