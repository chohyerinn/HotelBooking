# Library and Function Notes

A short reference for the libraries, classes, and helper functions used in the project, with the parameters we actually pass.

---

## Our top-level function: `compare_booking_models`

This is the open-source contribution. One function that takes a cleaned booking dataset and runs the full preprocessing + model sweep + final evaluation in one call, so the comparison code doesn't have to be repeated. Lives in `src/05_open_source_model_comparison.py`.

What it does, in order:

1. Draws a stratified sample and splits it into training and test sets.
2. Iterates over every combination of `{StandardScaler, MinMaxScaler}` × `{OneHotEncoder, OrdinalEncoder}` × the supported model/parameter settings (Logistic Regression, Decision Tree, KNN).
3. Scores each combination with stratified k-fold cross-validation on the training set only.
4. Ranks by mean F1, returns the top five.
5. Refits the best one on the full training set and scores it on the test set.

Parameters:

- `cleaned_data` — cleaned `pandas.DataFrame` with predictors and the target column.
- `target_column="is_canceled"` — name of the binary outcome column.
- `sample_size=20_000` — stratified sample size. `None` uses everything.
- `test_size=0.2` — fraction reserved for the final test set.
- `cv_splits=5` — number of stratified CV folds.
- `random_state=42` — seed for sampling, splitting, and tree models.

Returns a dict with `all_results`, `top_five`, `selected_combination`, `test_scores`, and the fitted `best_model` pipeline.

Minimal usage (the script filenames start with a digit, so we load them with `importlib` when calling from outside; inside the notebook the function is defined directly in a cell):

```python
import importlib.util, sys
from pathlib import Path

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

utils = load(Path("src/project_utils.py"), "project_utils")
comparison = load(
    Path("src/05_open_source_model_comparison.py"), "model_comparison"
)

output = comparison.compare_booking_models(utils.load_cleaned_data())
print(output["top_five"])
print(output["test_scores"])
```

---

## Building the preprocessing + model pipelines

### `ColumnTransformer`

Used to apply different preprocessing to numeric and categorical columns inside one object. Numeric columns go through median imputation + scaling; categorical columns go through most-frequent imputation + encoding. Without this we'd have to preprocess the two groups separately and stitch them back, which is exactly the kind of repetition we want to avoid.

Parameter we pass: `transformers` — a list of `(name, transformer, columns)` triples. In our case `"numeric"` points at the numeric `Pipeline` over `numerical_columns`, and `"categorical"` points at the categorical `Pipeline` over `categorical_columns`.

### `Pipeline`

Chains preprocessing steps and a model in a fixed order. The important property for us: during cross-validation, every step in the pipeline is fitted on the training fold only. That prevents validation data from sneaking into the imputer or scaler statistics.

Parameter we pass: `steps` — a list of `(name, transformer-or-model)` tuples. The preprocessing pipelines look like `[("imputer", ...), ("scaler", ...)]` or `[("imputer", ...), ("encoder", ...)]`. The full modeling pipeline is `[("preprocessor", preprocessor), ("model", model)]`.

### `SimpleImputer`

Fills missing values before the model sees them. After our cleaning step there shouldn't be many left, but a few categorical columns can still produce gaps after the train/test split, so we keep the imputer in the pipeline as a safety net.

Parameters:
- `strategy="median"` — for numeric columns. Median is more robust than mean against the long-tailed columns like `adr` and `lead_time`.
- `strategy="most_frequent"` — for categorical columns. Falls back to the modal category.

### `clone`

Returns a fresh copy of an estimator with the same parameter settings but no fitted state. We use it inside the comparison function so that each fold and each combination starts from a clean slate — otherwise the same scaler/encoder/model object would carry state from the previous iteration.

Parameter: `estimator` — the scaler, encoder, or model being copied.

---

## Encoding alternatives

### `OrdinalEncoder`

Tested as an alternative to one-hot encoding. Replaces each category with an integer code. Cheaper than one-hot when cardinality is high, but imposes an arbitrary ordering on the categories, which can hurt linear models. We let the comparison function decide.

Parameters:
- `handle_unknown="use_encoded_value"` — don't crash when a validation fold contains a category the encoder hasn't seen.
- `unknown_value=-1` — what to write for those unseen categories.

---

## Cross-validation and scoring

### `StratifiedKFold`

Splits the data into folds while keeping the cancellation/non-cancellation ratio roughly constant in each fold. Plain `KFold` doesn't guarantee that, and with a ~37/63 class split, an unlucky fold could be noticeably off-balance.

Parameters:
- `n_splits=5` (or whatever `cv_splits` is) — five validation rounds.
- `shuffle=True` — randomize row order before assigning folds.
- `random_state=…` — keeps the shuffled split reproducible.
- `.split(X, y)` — `y` is used here to preserve the class proportions.

### `balanced_accuracy_score`

Average of per-class recall. We report it alongside accuracy because plain accuracy gives partial credit for just predicting "not canceled" on every booking, given the class imbalance.

Parameters: `actual`, `prediction` — the true and predicted labels.

### `roc_auc_score`

Area under the ROC curve. Summarizes how well the model separates the two classes across all probability thresholds, independent of any single decision boundary.

Parameters:
- `actual` — true labels.
- `probability` — predicted probability for the positive (canceled) class, taken from `predict_proba(...)[:, 1]`.

### `predict_proba`

Returns class probabilities instead of hard labels. We need this for ROC-AUC and for the demo's risk thresholds.

- `X` — the feature data.
- `[:, 1]` — slice the column for the canceled class.

### `RocCurveDisplay`

Used to plot the ROC curves of the three baseline models on the same axes.

Parameters we pass to `from_predictions`:
- `actual` — true labels.
- `probability` — predicted probability for the canceled class.
- `name=model_name` — legend label.
- `ax=ax` — shared matplotlib axes so the three curves overlay.

### `feature_importances_`

Attribute (not a function) on a fitted Decision Tree. Gives a relative contribution score per feature. We use it to see which booking variables drive the tree's splits — useful for explaining the model in the presentation.

This is specific to tree-based models. Logistic Regression and KNN don't expose it, so for those we'd have to look at coefficients (for LR) or skip importance entirely (for KNN).

Reads as `(feature_name, importance_score)` pairs after we line it up with the preprocessor's column names.

---

## Clustering

### `silhouette_score`

A clustering quality measure: roughly, how close each point is to its own cluster compared to the next closest cluster. Higher is better. We use it alongside the elbow plot when picking the number of clusters for K-means.

Parameters:
- `X_scaled` — the standardized clustering features.
- `labels` — cluster assignments from K-means.
- `sample_size=10_000` — silhouette is `O(n²)` in distance computations, so we evaluate on a sample.
- `random_state=RANDOM_STATE` — makes the sample reproducible.

### `PCA`

Reduces the clustering features down to two components for plotting. K-means itself runs in the original feature space; PCA is only here so we can show the clusters on a scatter plot.

Parameters:
- `n_components=2` — two axes, PC1 and PC2.
- `.fit_transform(X_scaled)` — learn the projection and apply it in one call.

### `np.log1p`

Computes `log(1 + x)` element-wise. We apply it to right-skewed columns (`lead_time`, `adr`, `total_stays`) before standardization, so that a small number of extreme bookings don't dominate the Euclidean distances K-means uses. `log1p` instead of `log` because some of these columns contain zeros, and `log(0)` is undefined.

Applied to: the DataFrame of numeric clustering features.

---



