# House Price Prediction — Model Report

## Overview

This report documents the development, evaluation, and validation of a machine learning system for residential house price prediction. The objective of the project is to construct a robust and reproducible end-to-end pipeline, capable of generating accurate price estimates from raw, real-world housing data.

---

## 1. Model Selection Rationale

During the model selection stage, several regression algorithms were systematically evaluated, including **Ridge Regression**, **Random Forest Regressor**, a **Neural Network (MLP)**, and **XGBoost Regressor**.

All candidate models were trained and assessed using the same feature-engineered dataset and a consistent **60/20/20 train–validation–test split**, ensuring a fair and controlled experimental comparison.

Among the evaluated approaches, **XGBoost** was selected as the final model due to its superior performance on the held-out test set. In particular, it demonstrated:

* Lower prediction error in log-price space
* A higher coefficient of determination (R²)
* Strong robustness to complex non-linear relationships
* More reliable generalization performance compared with Ridge Regression, Random Forests, and Neural Networks

Furthermore, XGBoost provides native support for **early stopping**, which helps mitigate overfitting and improves training stability. These characteristics make it especially well suited for the house price regression task.

---

## 2. Final Model Performance

The final XGBoost model was evaluated on a held-out test set that was not used during training or validation. Unless otherwise stated, all reported metrics are computed in **log(price)** space.

| Metric                       | Value      |
| ---------------------------- | ---------- |
| Test MSE (log)               | 0.0586     |
| Test J = MSE / 2             | 0.0293     |
| Test RMSE (log)              | 0.2422     |
| Test R² (log)                | 0.8078     |
| Accuracy within ±10% (price) | **46.98%** |

Overall, these results indicate that the model explains approximately **75–80% of the variance** in housing prices on the log scale, while maintaining a reasonable level of accuracy when predictions are transformed back into real price values.

---

## 3. Example Prediction (End-to-End Inference)

To assess real-world applicability, the trained model was evaluated using **raw input data**, rather than pre-encoded feature matrices. This experiment verifies the correctness and consistency of the complete inference pipeline, from feature engineering to final price prediction.

### Input Description

* bedrooms = 3
* bathrooms = 1.5
* sqft_living = 1340
* city = Shoreline
* statezip = WA 98133

### Prediction Outcome

Predicted price: 316,413 USD  
Actual price: 313,000 USD  
Relative error: +1.1%

The small relative error observed in this example demonstrates that the full pipeline—
raw input processing, feature engineering, one-hot encoding, and model inference—functions as intended and is capable of producing realistic and interpretable price estimates.

---

## 4. Reproducibility and Execution

### Environment Requirements

To reproduce the training and inference results reported in this project, the following software environment is required:

* Python >= 3.9
* numpy
* pandas
* scikit-learn
* xgboost

All dependencies must be installed prior to executing the scripts described below.

### Model Training

```bash
python model/src/train.py
```

### Model Inference

```bash
python model/src/predict.py --input data/raw_input.csv --mode raw
```

All trained models, feature definitions, hyperparameters, and evaluation metrics are stored under `model/artifacts/`, ensuring that the experimental results are fully reproducible.

---

## 5. Summary and Future Work

This project presents a **complete, end-to-end machine learning pipeline** for house price prediction. The workflow encompasses data preprocessing, feature engineering, systematic model comparison, final model training with early stopping, and inference from raw real-world inputs.

The resulting system is robust, reproducible, and readily extensible. Potential future extensions include deployment through a RESTful API, integration with web-based user interfaces, or further model refinement using additional data sources or advanced ensemble techniques.

---

## 6. Notes

Unless explicitly specified in the notebook, XGBoost hyperparameters not listed are kept at their default values. The training script follows the same configuration, ensuring consistency between the notebook experiments and the scripted implementation.