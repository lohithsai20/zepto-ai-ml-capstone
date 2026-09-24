# Modeling Results

| classifier          |   accuracy |   precision |   recall |       f1 |      auc |
|:--------------------|-----------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.808989 |    0.783333 | 0.691176 | 0.734375 | 0.860963 |
| Decision Tree       |   0.764045 |    0.76     | 0.558824 | 0.644068 | 0.837366 |
| Random Forest       |   0.808989 |    0.765625 | 0.720588 | 0.742424 | 0.819586 |

## Regression

| model             |     MAE |    RMSE |       R2 |   Adjusted_R2 |
|:------------------|--------:|--------:|---------:|--------------:|
| Linear Regression | 18.3735 | 41.2921 | 0.360916 |      0.260668 |

## Final recommendation

Model selection was based on the reported test-set F1, while also considering accuracy, recall and AUC. The selected classifier is Random Forest, with accuracy=0.809, precision=0.766, recall=0.721, F1=0.742, and AUC=0.820. These values should be read together because no single classification metric captures every operational trade-off. The complete preprocessing-plus-estimator pipeline was saved and successfully reloaded against raw feature columns. The fare regression metrics are reported separately because regression and classification metrics are different quantities.

## Imbalance comparison

| strategy              |   precision |   recall |       f1 |
|:----------------------|------------:|---------:|---------:|
| baseline              |    0.783333 | 0.691176 | 0.734375 |
| class_weight_balanced |    0.71831  | 0.75     | 0.733813 |
| smote                 |    0.735294 | 0.735294 | 0.735294 |

## Tuning

Best RF parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 200}`

OOB score: `0.8213783403656821`

Residual interpretation: the residual plot should be inspected for a funnel or systematic spread; the script also prints a simple correlation-based indication, which is supporting evidence rather than a formal heteroscedasticity test.
