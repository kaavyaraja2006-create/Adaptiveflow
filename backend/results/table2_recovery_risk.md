### Table 2. Next-day recovery-risk forecasting on Itemlet (mean ± std over folds)

| Model | MAE | RMSE | R2 | Precision | Recall | F1 | Accuracy | ROC-AUC | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Predict-mean | 0.244 ± 0.021 | 0.278 ± 0.021 | -0.192 ± 0.177 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.702 ± 0.143 | 0.500 ± 0.000 | 0.000 ± 0.000 |
| Persistence (today's risk) | 0.068 ± 0.003 | 0.121 ± 0.003 | 0.765 ± 0.075 | 0.824 ± 0.063 | 0.825 ± 0.063 | 0.825 ± 0.063 | 0.910 ± 0.029 | 0.943 ± 0.009 | 0.001 ± 0.000 |
| Ridge regression | 0.098 ± 0.003 | 0.123 ± 0.004 | 0.759 ± 0.063 | 0.797 ± 0.048 | 0.697 ± 0.282 | 0.703 ± 0.235 | 0.888 ± 0.026 | 0.948 ± 0.015 | 0.518 ± 0.405 |
| Random Forest | 0.077 ± 0.004 | 0.102 ± 0.003 | 0.834 ± 0.054 | 0.835 ± 0.036 | 0.711 ± 0.161 | 0.761 ± 0.118 | 0.895 ± 0.028 | 0.958 ± 0.011 | 29.977 ± 20.552 |
| Gradient Boosting | 0.056 ± 0.002 | 0.083 ± 0.002 | 0.890 ± 0.029 | 0.860 ± 0.038 | 0.818 ± 0.059 | 0.838 ± 0.049 | 0.918 ± 0.028 | 0.967 ± 0.009 | 16.775 ± 15.700 |
| BiGRU (AdaptiveFlow) | 0.138 ± 0.028 | 0.168 ± 0.033 | 0.559 ± 0.131 | 0.781 ± 0.077 | 0.651 ± 0.260 | 0.670 ± 0.212 | 0.864 ± 0.054 | 0.922 ± 0.031 | 1.593 ± 1.676 |

_5-fold CV, whole developers held out; 58 developers. Task: predict next-day recovery risk from a 14-day window. 'Elevated risk' = next-day risk >= 0.55 (mean positive rate 29.8%). Regression metrics are on the 0-1 risk score._
