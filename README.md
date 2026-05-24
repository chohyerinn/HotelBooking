# Hotel Booking Cancellation Prediction

## Project Objective

This project aims to predict hotel booking cancellations and identify patterns associated with cancellation behavior. Cancellations can negatively affect hotel revenue and operational efficiency by creating uncertainty in room allocation, pricing, and overbooking decisions. By estimating cancellation risk in advance, hotels can improve reservation management and support more effective operational decision-making.

In addition, this project applies clustering analysis to identify distinct booking groups based on reservation characteristics, providing further insights into customer behavior and cancellation risk.

## Dataset

We use the **Hotel Booking Demand** dataset, which contains reservation records
from a City Hotel and a Resort Hotel in Portugal.

- Number of records: 119,390
- Number of columns: 32
- Prediction target: `is_canceled`
  - `0`: the booking was not canceled
  - `1`: the booking was canceled

The raw dataset includes numerical and categorical features, as well as missing values, duplicate rows, and a few suspicious records that need to be examined during preprocessing.

Sources:

- Kaggle: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Original publication: Antonio, Almeida, and Nunes (2019),
  [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126)

## Analysis Plan

The main analysis task is a binary classification problem with `is_canceled` as the target variable. We plan to compare Logistic Regression, Decision Tree, and K-Nearest Neighbors models using k-fold cross validation. Since identifying bookings that may actually be canceled is important for hotel operations, recall and F1-score will be considered especially important, together with accuracy, precision, and a confusion matrix.

As an additional analysis, K-means clustering will be used to explore whether meaningful booking groups can be found and interpreted from a business perspective.

The project will proceed as follows:

1. Explore the dataset, including distributions, missing values, duplicates, unusual values, and cancellation patterns.
2. Clean and preprocess the data, create useful features, encode categorical variables, and scale numerical variables when needed.
3. Train and compare the classification models, then evaluate their results.
4. Apply K-means clustering and interpret the characteristics of the identified booking groups.

Because the purpose is to predict cancellation before the final booking outcome is known, post-outcome information such as `reservation_status` and `reservation_status_date` will not be used as model inputs.
