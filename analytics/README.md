# Analytics Pipeline

## Run order

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

`01_eda.py` performs the one network/cache load with `sns.load_dataset('titanic')` when `titanic.csv` does not exist, immediately saves the raw DataFrame as `titanic.csv`, and writes `cleaned_titanic.csv`. `02_modeling.py` reads the saved CSV and never calls `sns.load_dataset`.

## Required decisions

Missingness is measured before cleaning. Under 5% missingness is handled by dropping affected rows; 5%–30% is imputed; above 30% is dropped when imputation is unreliable. The script prints the exact percentages and decisions. `deck` is the high-missingness column and is dropped. Modeling uses its own train-only imputation, encoding, and scaling inside a `ColumnTransformer`/Pipeline.

The modeling split is stratified before preprocessing so train and test retain similar survival proportions. SMOTE is applied only inside an imbalanced-learn pipeline on the training fold. Random Forest tuning uses `oob_score=True` so the best estimator exposes an OOB score.

The final `best_pipeline.joblib` contains preprocessing and the estimator together and is tested after reload using raw, unprocessed feature columns.

**Important for submission:** after the first successful run, commit `titanic.csv`, generated result markdown/CSV files, plots, and `best_pipeline.joblib` as required by the assignment.
