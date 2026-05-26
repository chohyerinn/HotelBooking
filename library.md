# Additional Library and Function Explanations

This document explains the additional Python and scikit-learn tools used in the project, including the parameters applied in the source code.

## `ColumnTransformer`

`ColumnTransformer` was used in the classification modeling and model comparison steps.

`ColumnTransformer` applies different preprocessing steps to selected groups of columns within one object. It is used because numeric booking fields require imputation and scaling, while categorical booking fields require imputation and encoding.

Parameters used:

* `transformers`: a list of `(name, transformer, columns)` tuples.
* `"numeric"`: applies a numeric `Pipeline` to `numerical_columns`.
* `"categorical"`: applies a categorical `Pipeline` to `categorical_columns`.

## `Pipeline`

`Pipeline` was used to connect preprocessing and classification during model evaluation.

`Pipeline` connects preprocessing steps and a model in a fixed order. In cross-validation, the preprocessing steps are fitted on each training fold only, which helps prevent data leakage from validation data.

Parameters used:

* `steps`: a list of `(name, transformer or model)` tuples.
* Preprocessing pipelines use `("imputer", ...)`, followed by `("scaler", ...)` or `("encoder", ...)`.
* Modeling pipelines use `("preprocessor", preprocessor)` followed by `("model", model)`.

## `SimpleImputer`

`SimpleImputer` was used as part of the preprocessing steps for classification models.

`SimpleImputer` replaces remaining missing values before a model is fitted. This allows every transformed row to be passed into scikit-learn models without missing-value errors.

Parameters used:

* `strategy="median"`: fills missing numeric values with the median of the training data.
* `strategy="most_frequent"`: fills missing categorical values with the most frequent training value.

## `StratifiedKFold`

`StratifiedKFold` was used to evaluate classification models consistently.

`StratifiedKFold` creates cross-validation folds while preserving approximately the same cancellation and non-cancellation ratio in every fold. Regular `KFold` splits rows without guaranteeing this class balance.

Parameters used:

* `n_splits=5` or `n_splits=cv_splits`: divides data into five validation rounds by default.
* `shuffle=True`: randomizes rows before assigning folds.
* `random_state`: keeps the shuffled split reproducible.

Method used:

* `split(X, y)`: returns training and test indices while using `y` to preserve class proportions.

## `balanced_accuracy_score`

`balanced_accuracy_score` was used when comparing classification performance.

`balanced_accuracy_score` evaluates a classifier by averaging recall for each class. It is useful here because canceled and non-canceled bookings do not occur in exactly equal proportions.

Parameters used:

* `actual`: true cancellation labels.
* `prediction`: predicted cancellation labels.

## `RocCurveDisplay`

`RocCurveDisplay` was used to compare the classification models visually.

`RocCurveDisplay` visualizes the trade-off between true positive rate and false positive rate across classification thresholds.

Method and parameters used:

* `from_predictions(actual, probability, name=model_name, ax=ax)`
* `actual`: true labels.
* `probability`: predicted probabilities for the canceled class.
* `name`: model label shown in the legend.
* `ax`: matplotlib axes on which multiple model curves are compared.

## `silhouette_score`

`silhouette_score` was used when selecting an appropriate number of clusters.

`silhouette_score` measures whether samples are close to their assigned cluster and separated from other clusters. Larger scores indicate clearer clustering structure.

Parameters used:

* `X_scaled`: standardized clustering features.
* `labels`: cluster labels predicted by K-means.
* `sample_size=10_000`: evaluates a sample to reduce computation time on the full dataset.
* `random_state=RANDOM_STATE`: makes that sample reproducible.

## `PCA`

`PCA` was used to visualize booking clusters in two dimensions.

`PCA` reduces several numeric clustering features to a smaller set of components that retain major variation in the data. Here it is used only to display clusters in a two-dimensional scatter plot.

Parameters and method used:

* `n_components=2`: creates two plotted axes, `PC1` and `PC2`.
* `fit_transform(X_scaled)`: learns the projection from scaled features and transforms the rows.

## `OrdinalEncoder`

`OrdinalEncoder` was tested as an alternative categorical encoding method.

`OrdinalEncoder` converts each categorical value to a numeric code. It is evaluated as an alternative to one-hot encoding in the model comparison function.

Parameters used:

* `handle_unknown="use_encoded_value"`: allows unseen validation categories to be encoded.
* `unknown_value=-1`: represents categories that were not present when the encoder was fitted.

## `np.log1p`

`np.log1p` was used to compress skewed numeric features before clustering.

`np.log1p(x)` computes `log(1 + x)` element-wise, which reduces the influence of very large values while still allowing zeros. It is used because clustering features such as `lead_time`, `adr`, and `total_stays` are right-skewed, and without compression a few extreme bookings would dominate the distance calculation in K-means.

Parameters used:

* `df[cluster_features]`: a DataFrame of numeric clustering features; the transformation is applied element-wise before standardization.

## `roc_auc_score`

`roc_auc_score` was used as an additional evaluation metric alongside `RocCurveDisplay`.

`roc_auc_score` returns the area under the ROC curve, summarizing how well a model separates canceled from non-canceled bookings across all probability thresholds. A score of 0.5 indicates random guessing and 1.0 indicates perfect separation.

Parameters used:

* `actual`: true cancellation labels.
* `probability`: predicted probability for the canceled class, obtained from `predict_proba`.
