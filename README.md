# HIT140 Assessment 3 – Darwin Group 65

## Objective 2: FIFA World Cup 2026 Predictive Modelling

### Group Members
- Muhammad Kumail Abbas – S397291
- Roman Bisural – S397541
- Suriya Sankar – S397448
- Khushi – S398777

## Project Aim
This repository contains the Python code, data, outputs, figures and collaboration evidence for HIT140 Assessment 3, Objective 2.

Two regression tasks are completed:

1. **Regression 2.1 – Goal difference**
   - 104 match observations
   - 8 pre-match explanatory variables
   - 82 training rows / 22 held-out test rows
   - Final CV-selected model: **Lasso, alpha = 1**

2. **Regression 2.2 – Goals scored**
   - 208 team-match observations from 104 matches
   - 8 pre-match explanatory variables
   - 164 training rows / 44 held-out test rows
   - Final CV-selected model: **Ridge, alpha = 100**

## Predictors
Four historical tournament variables are calculated for each side:

- prior average goals scored
- prior average goals conceded
- prior points per game
- prior clean-sheet rate

The model therefore uses exactly eight pre-match explanatory variables.

## Leakage Safeguards
Predictors are created only from information available before the target match. Same-date and current-match updates are deferred so the target match cannot influence its own predictors.

## Models Compared
- Training-mean baseline
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression

Model selection uses expanding chronological cross-validation. The held-out test period is used only for final descriptive evaluation and not for tuning.

## Final Results

### Regression 2.1
| Model | Alpha | CV MAE | CV MAE SD | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | – | 1.8286 | 0.2410 | 1.4656 | 1.7678 | -0.1662 |
| OLS | – | 3.3503 | 2.4099 | 1.2573 | 1.6960 | -0.0734 |
| Ridge | 100 | 1.8892 | 0.4158 | 1.3034 | 1.6574 | -0.0251 |
| Lasso | 1 | 1.8286 | 0.2410 | 1.4656 | 1.7678 | -0.1662 |

**Primary model:** Lasso, alpha = 1

### Regression 2.2
| Model | Alpha | CV MAE | CV MAE SD | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | – | 1.2316 | 0.0759 | 1.0654 | 1.3238 | -0.0016 |
| OLS | – | 1.4800 | 0.5333 | 0.9637 | 1.3209 | 0.0027 |
| Ridge | 100 | 1.2105 | 0.1631 | 0.9883 | 1.3005 | 0.0333 |
| Lasso | 0.1 | 1.2149 | 0.1496 | 0.9997 | 1.3087 | 0.0211 |

**Primary model:** Ridge, alpha = 100

## Robustness Check
Training-only sensitivity analysis confirms:

- Regression 2.1:
  - Ridge alpha=1000 -> CV MAE 1.8360
  - Lasso alpha=1 -> CV MAE 1.8286
  - Primary model remains Lasso alpha=1

- Regression 2.2:
  - Ridge alpha=100 -> CV MAE 1.2105
  - Lasso alpha=0.1 -> CV MAE 1.2149
  - Primary model remains Ridge alpha=100

No held-out test outcomes are used for sensitivity tuning.

## Run Order
Run the scripts in this order:

1. `05_objective2_prepare_datasets_CORRECTED.py`
2. `06_objective2_linear_regression_2_1.py`
3. `07_objective2_linear_regression_2_2.py`
4. `08_objective2_eda.py`
5. `09_objective2_model_comparison.py`
6. `10_objective2_hd_diagnostics.py`

## Repository Structure
```text
HIT140-Assessment-3-Darwin-Group-65/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Assigment 3/
│   ├── 05_objective2_prepare_datasets_CORRECTED.py
│   ├── 06_objective2_linear_regression_2_1.py
│   ├── 07_objective2_linear_regression_2_2.py
│   ├── 08_objective2_eda.py
│   ├── 09_objective2_model_comparison.py
│   ├── 10_objective2_hd_diagnostics.py
│  
│ 
│
├── DATA/
└── OUTPUT/
    └── OBJECTIVE2/
```

## Collaboration
Group collaboration was documented primarily through Microsoft Teams meetings, agreed task allocation, individual coding responsibilities, repository commits, independent review of another member's work, and final group quality checks.
The coding responsibilities were divided as follows:
- Muhammad Kumail Abbas (S397291): data/source validation, missing-value checks, exploratory data analysis and correlation analysis, with independent review of Regression 2.2 results and methodology.
- Roman Bisural (S397541): leakage-safe dataset preparation, Regression 2.1 modelling, chronological train/test splitting, expanding cross-validation and model comparison, with review of Regression 2.2 robustness and reproducibility.
- Suriya Sankar (S397448): Regression 2.2 modelling, evaluation and interpretation, with independent review of Regression 2.1 cross-validation stability and limitations.
- Khushi (S398777): training-only robustness and sensitivity analysis, figure review, cross-task synthesis and final quality assurance.
Three Microsoft Teams meetings were used to document planning, progress review, model results, robustness checks and final submission readiness.
