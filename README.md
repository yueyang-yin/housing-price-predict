# House Price Prediction

This repository contains an **end-to-end machine learning practice project** for residential house price prediction. The primary goal of the project is to practice data preprocessing, feature engineering, model selection, and basic ML engineering workflows.

> ⚠️ **Disclaimer**
> This project is created **for learning and practice purposes only**.
> The models, results, and conclusions presented here **should not be used as a reference for real-world decision-making, investment analysis, or commercial applications**.

---

## Project Overview

The project covers the complete pipeline of a typical supervised learning task:

* Data cleaning and exploratory analysis
* Feature engineering and transformation
* Model comparison and selection
* Final model training using XGBoost with early stopping
* End-to-end inference from raw input data

A detailed technical report describing the modeling process, evaluation metrics, and example predictions is provided separately.

📄 **Full technical report**: [report/README.md](report/README.md)

---

## Dataset

The dataset used in this project is sourced from Kaggle:

* **House price prediction** dataset
* Platform: Kaggle
* Purpose: Educational and exploratory analysis

🔗 **Dataset source**:
[https://www.kaggle.com/datasets/shree1992/housedata](https://www.kaggle.com/datasets/shree1992/housedata)

Please refer to the Kaggle page for the original dataset description, context, and licensing information.

---

## Reproducibility (Quick Start)

To reproduce the training and inference results locally, ensure that the required environment is installed. For detailed environment requirements, please refer to [report/README.md](report/README.md).

### Train the model

```bash
python model/src/train.py
```

### Run prediction

```bash
python model/src/predict.py --input data/raw_input.csv --mode raw
```

All trained models, feature definitions, and evaluation metrics are stored under `model/artifacts/`.

---

## Repository Structure

```text
housing_price_predict_project/
├── data/            # Raw and processed datasets
├── model/           # Training and inference scripts
├── report/          # Detailed technical report
├── README.md        # Project overview (this file)
├── LICENSE
└── .gitignore
```

---

## Notes

This repository is intended to demonstrate **machine learning workflow and engineering practices**, rather than to provide a production-ready solution. Future extensions may include API deployment, additional model experimentation, or improved data validation.
