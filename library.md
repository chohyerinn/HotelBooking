# Additional Library and Function Explanations

This document explains the additional Python and scikit-learn tools used in the project, including the parameters applied in the source code.

## Open Source Contribution Function: `compare_booking_models`

`compare_booking_models` is the single public entry point for the reusable model-comparison contribution in `src/05_ModelComparison.py`. Given a cleaned booking dataset, it completes the full evaluation workflow without requiring users to repeat preprocessing, training, model selection, or testing code.

The function performs the following tasks:

1. Draws a stratified sample and makes a final training/test split.
2. Compares combinations of `StandardScaler` and `MinMaxScaler`, `OneHotEncoder` and `OrdinalEncoder`, and the supported Logistic Regression, Decision Tree, and K-Nearest Neighbors parameter settings.
3. Evaluates every combination using stratified k-fold cross-validation on the training set only.
4. Ranks the combinations by mean F1-score and balanced accuracy and returns the top five.
5. Fits the best-ranked pipeline on the full training set and evaluates it once on the held-out test set.

Parameters used:

* `cleaned_data`: cleaned `pandas.DataFrame` containing predictors and `is_canceled`.
* `target_column="is_canceled"`: binary outcome column to predict.
* `sample_size=20_000`: number of stratified observations used; set to `None` to use all rows.
* `test_size=0.2`: held-out proportion used for final testing.
* `cv_splits=5`: number of stratified cross-validation folds.
* `random_state=42`: seed for repeatable sampling, splitting, and tree fitting.

Returned values:

* `all_results`: cross-validation scores for all preprocessing/model combinations.
* `top_five`: the five highest-ranked combinations.
* `selected_combination`: the selected configuration and its cross-validation scores.
* `test_scores`: final held-out accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC.
* `best_model`: the fitted scikit-learn pipeline for the selected configuration.

Example:

```python
import importlib.util
import sys

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location(
    "model_comparison", "src/05_ModelComparison.py"
)
model_comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_comparison)

output = model_comparison.compare_booking_models(
    model_comparison.load_cleaned_data()
)
print(output["top_five"])
print(output["test_scores"])
```

## `ColumnTransformer`

`ColumnTransformer` was used in the classification modeling and model comparison steps.

`ColumnTransformer` applies different preprocessing steps to selected groups of columns within one object. It is used because numeric booking fields require imputation and scaling, while categorical booking fields require imputation and encoding.

* `transformers`: a list of `(name, transformer, columns)` tuples.
* `"numeric"`: applies a numeric `Pipeline` to `numerical_columns`.
* `"categorical"`: applies a categorical `Pipeline` to `categorical_columns`.

## `Pipeline`

`Pipeline` was used to connect preprocessing and classification during model evaluation.

`Pipeline` connects preprocessing steps and a model in a fixed order. In cross-validation, the preprocessing steps are fitted on each training fold only, which helps prevent data leakage from validation data.

* `steps`: a list of `(name, transformer or model)` tuples.
* Preprocessing pipelines use `("imputer", ...)`, followed by `("scaler", ...)` or `("encoder", ...)`.
* Modeling pipelines use `("preprocessor", preprocessor)` followed by `("model", model)`.

## `SimpleImputer`

`SimpleImputer` was used as part of the preprocessing steps for classification models.

`SimpleImputer` replaces remaining missing values before a model is fitted. This allows every transformed row to be passed into scikit-learn models without missing-value errors.

* `strategy="median"`: fills missing numeric values with the median of the training data.
* `strategy="most_frequent"`: fills missing categorical values with the most frequent training value.

## `StratifiedKFold`

`StratifiedKFold` was used to evaluate classification models consistently.

`StratifiedKFold` creates cross-validation folds while preserving approximately the same cancellation and non-cancellation ratio in every fold. Regular `KFold` splits rows without guaranteeing this class balance.

* `n_splits=5` or `n_splits=cv_splits`: divides data into five validation rounds by default.
* `shuffle=True`: randomizes rows before assigning folds.
* `random_state`: keeps the shuffled split reproducible.

* `split(X, y)`: returns training and validation indices while using `y` to preserve class proportions.

## `clone`

`clone` was used when creating pipelines for model comparison.

`clone` creates a new estimator with the same parameter settings but without previously learned results. This allows each scaling, encoding, and model combination to be evaluated independently during cross-validation.

* `estimator`: the scaler, encoder, or classification model copied before fitting.

## `balanced_accuracy_score`

`balanced_accuracy_score` was used when comparing classification performance.

`balanced_accuracy_score` evaluates a classifier by averaging recall for each class. It is useful here because canceled and non-canceled bookings do not occur in exactly equal proportions.

* `actual`: true cancellation labels.
* `prediction`: predicted cancellation labels.

## `RocCurveDisplay`

`RocCurveDisplay` was used to compare the classification models visually.

`RocCurveDisplay` visualizes the trade-off between true positive rate and false positive rate across classification thresholds.

* `from_predictions(actual, probability, name=model_name, ax=ax)`
* `actual`: true labels.
* `probability`: predicted probabilities for the canceled class.
* `name`: model label shown in the legend.
* `ax`: matplotlib axes on which multiple model curves are compared.

## `silhouette_score`

`silhouette_score` was used when selecting an appropriate number of clusters.

`silhouette_score` measures whether samples are close to their assigned cluster and separated from other clusters. Larger scores indicate clearer clustering structure.

* `X_scaled`: standardized clustering features.
* `labels`: cluster labels predicted by K-means.
* `sample_size=10_000`: evaluates a sample to reduce computation time on the full dataset.
* `random_state=RANDOM_STATE`: makes that sample reproducible.

## `PCA`

`PCA` was used to visualize booking clusters in two dimensions.

`PCA` reduces several numeric clustering features to a smaller set of components that retain major variation in the data. Here it is used only to display clusters in a two-dimensional scatter plot.

* `n_components=2`: creates two plotted axes, `PC1` and `PC2`.
* `fit_transform(X_scaled)`: learns the projection from scaled features and transforms the rows.

## `OrdinalEncoder`

`OrdinalEncoder` was tested as an alternative categorical encoding method.

`OrdinalEncoder` converts each categorical value to a numeric code. It is evaluated as an alternative to one-hot encoding in the model comparison function.

* `handle_unknown="use_encoded_value"`: allows unseen validation categories to be encoded.
* `unknown_value=-1`: represents categories that were not present when the encoder was fitted.

## `np.log1p`

`np.log1p` was used to compress skewed numeric features before clustering.

`np.log1p(x)` computes `log(1 + x)` element-wise, which reduces the influence of very large values while still allowing zeros. It is used because clustering features such as `lead_time`, `adr`, and `total_stays` are right-skewed, and without compression a few extreme bookings would dominate the distance calculation in K-means.

* `df[cluster_features]`: a DataFrame of numeric clustering features; the transformation is applied element-wise before standardization.

## `roc_auc_score`

`roc_auc_score` was used in model comparison and in the final test set evaluation.

`roc_auc_score` returns the area under the ROC curve, summarizing how well a model separates canceled from non-canceled bookings across all probability thresholds. A score of 0.5 indicates random guessing and 1.0 indicates perfect separation.

* `actual`: true cancellation labels.
* `probability`: predicted probability for the canceled class, obtained from `predict_proba`.

## `feature_importances_`

`feature_importances_` was used to interpret the Decision Tree model.

`feature_importances_` provides the relative contribution of each feature to the model's prediction decisions. In this project, it is used to identify which booking variables have the greatest influence on cancellation prediction.

* `feature`: the name of each input variable after preprocessing.
* `importance`: the contribution score assigned to each feature by the fitted Decision Tree model.

## `predict_proba`

`predict_proba` was used when evaluating classification models.

`predict_proba` returns the estimated probability for each class instead of only returning a predicted label. In this project, the predicted probability of class `1` is used to measure cancellation risk and calculate ROC-AUC.

* `X`: the booking feature data used to generate predicted class probabilities.

* `[:, 1]`: selects the predicted probability that a booking is canceled.
