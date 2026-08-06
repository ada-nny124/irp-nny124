# Model Inventory

- Date generated: `2026-07-29`
- Scope: models trained or benchmarked in this repo, with the best tracked saved performance for each pipeline.

## Baseline models

| Model name | Performance | Why this model is relevant to this research | Notes |
| --- | --- | --- | --- |
| `logistic regression` | balanced accuracy `0.849219`, ROC AUC `0.925031` | Provides a simple, interpretable baseline for whether the bound-mass outcome is reasonably approximated by a linear decision boundary. | Useful for establishing a transparent starting point. |
| `random forest` | `R² = 0.897127`, `MAE = 0.018394` | Captures nonlinear interactions in a flexible but still interpretable tree-based baseline. | Strong benchmark for comparing against more specialized models. |
| `gradient boosting` | `R² = 0.884015`, `MAE = 0.022750` | Serves as a strong nonlinear baseline that often performs well on tabular scientific data. | Helps test whether added physics structure is actually giving a meaningful gain. |
| `random forest with physics-structured features` | `R² = 0.9225`, `MAE = 0.0179` | Directly tests the central hypothesis that physically informed descriptors improve predictive performance beyond a standard tree baseline. | This is the current promoted physics-structured surrogate. |
| `gradient boosting with physics-structured features` | not separately tracked in saved metrics | Extends the strong boosting baseline with the same physics-structured feature set to test whether the gain comes from the physics features rather than the algorithm alone. | Kept as the natural physics-informed extension of the boosting baseline. |

## Advanced models

| Model name | Performance | Justification | Notes |
| --- | --- | --- | --- |
| `Hurdle NGBoost surrogate` | `R² = 0.948477`, `MAE = 0.012689`, `RMSE = 0.021121` | Strongest saved full-target surrogate in this set and directly relevant because it combines a physics-structured feature representation with a probabilistic hurdle framework. | Current best saved full-target BMF model. |
| `Two-stage CatBoost hurdle model` | `R² = 0.948321`, `MAE = 0.012170`, `RMSE = 0.021153` | A strong alternative to the NGBoost hurdle model that is highly relevant for capturing the zero-heavy structure of the target while retaining strong predictive accuracy. | Best first-pass candidate. |
| `XGBoost regressor` | `R² = 0.937662`, `MAE = 0.013222` | Relevant as a high-performing boosted-tree benchmark that can test whether a strong modern regressor benefits from the same physics-structured inputs. | Strongest single-stage booster. |
| `Regime-aware mixture of experts` | `R² = 0.931545`, `MAE = 0.015624`, `RMSE = 0.024345` | Relevant because it explicitly models regime-dependent behavior, which is important if the underlying process changes across physical regimes. | Strong regime model, below the hurdle winners. |
| `CatBoost regressor` | `R² = 0.930196`, `MAE = 0.015014` | Relevant as a strong native categorical-boosting baseline that can exploit structured tabular data well. | Strong native categorical booster. |
| `Hurdle beta surrogate` | `R² = 0.907283`, `MAE = 0.017284`, `RMSE = 0.028333` | Relevant because it tests a probabilistic response model for bounded targets, which is scientifically meaningful for a fraction-like output. | Statistically interesting, weaker than the hurdle winners. |
| `Multi-output GP surrogate` | `R² = 0.354815`, `MAE = 0.062443`, `RMSE = 0.074740` | Less competitive here, but still relevant as a structured probabilistic approach that can model correlations across related targets. | Not competitive on this archive. |
