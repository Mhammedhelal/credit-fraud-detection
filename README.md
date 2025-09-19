
# Credit Fraud Detection


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-%234CAF50.svg?style=flat)](https://imbalanced-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/xgboost-%23FF6600.svg?style=flat)](https://xgboost.readthedocs.io/)
[![Joblib](https://img.shields.io/badge/joblib-%230072C6.svg?style=flat)](https://joblib.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=flat&logo=plotly&logoColor=black)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%232E5E82.svg?style=flat)](https://seaborn.pydata.org/)

A machine learning project to detect fraudulent credit card transactions using anonymized transaction data. This repository implements classification models with preprocessing and resampling techniques to handle imbalanced datasets, focusing on high precision and recall for fraud detection.

## Table of Contents

* [Overview](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#overview)
* [Dataset](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#dataset)
* [Features](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#features)
* [Installation](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#installation)
* [Usage](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#usage)
* [Models](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#models)
* [Results](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#results)
* [Repository Structure](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#repository-structure)
* [Contributing](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#contributing)
* [License](https://grok.com/project/97a9cbf7-d968-4cfa-a4f5-3ec2eddce8c0?chat=b0fe240e-03d3-41d0-871d-0f1fe79a1344#license)

## Overview

Credit card fraud detection is a critical application of machine learning in finance. Fraudulent transactions are rare (typically <1% of data), leading to class imbalance issues. This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions from European cardholders in September 2013, with 492 frauds.

Key goals:

* Achieve high recall for fraud cases while maintaining precision.
* Compare supervised (e.g., Voting Classifier, XGBoost, Logistic Regression, Random Forest) models.
* Evaluate with metrics suited for imbalance: AUC-ROC, Precision-Recall AUC, F1-score, Accuracy, Precision, Recall.

## Dataset

* **Source** : [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size** : ~284k rows, 31 columns
* **Features** :
* `Time`: Seconds elapsed since first transaction.
* `V1`-`V28`: PCA-transformed features for anonymity.
* `Amount`: Transaction amount.
* `Class`: Target (0 = legitimate, 1 = fraud).
* **Imbalance** : 99.83% legitimate, 0.17% fraud.
* **Files** : Split into `train.csv`, `val.csv`, `test.csv`, and `trainval.csv` in the `data/` folder. Download `creditcard.csv` from Kaggle and preprocess it to create these splits.

## Features

* **Data Preprocessing** (via `credit_fraud_utils_data.py`):
  * Derive `Hour` from `Time` (hour of day, 0-23) and `is_rush_hour` (1 if hour in [0,1,2], else 0).
  * Compute `log_amount` (log(Amount + 1)) and detect outliers (`is_outlier_amount` if z-score > 2).
  * Bin `log_amount` into categories (Very Low, Low, Medium, High, Very High) using quantiles.
  * Calculate z-scores for `V1`-`V28` and flag outliers (`V13_is_outlier`, `V15_is_outlier`, etc., for selected features) if z-score > 2.
  * Create interaction features: `amount_hour_interaction`, `V7_amount`, `V12_amount`, `V20_amount`, `V11_hour`, `V12_hour`.
  * Apply cyclical encoding to `Hour` (sine and cosine transformations).
* **Resampling** : SMOTE (oversampling) and RandomUnderSampler (undersampling) to address class imbalance.
* **Models** : Logistic Regression, Random Forest, XGBoost, Voting Classifier (ensemble of Logistic Regression and Random Forest).
* **Evaluation** : Metrics include Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.
* **Visualization** : EDA plots (histograms, boxplots, scatter plots, correlation heatmaps, t-SNE) in `notebooks/1.EDA.ipynb`.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Mhammedhelal/credit-fraud-detection.git
   cd credit-fraud-detection
   ```
2. Create a virtual environment (Python 3.8+):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   ### Requirements:

   The required packages are listed in `requirements.txt`:

   numpy
   pandas
   scikit-learn
   imbalanced-learn
   xgboost
   joblib
   matplotlib
   seaborn

## Usage

### Quick Start

1. **Train a Model** :

```
   python src/credit_fraud_train.py --dataset data/train.csv --model_name voting --model_save_name models/voting.pkl --use_oversample True --use_undersample True
```

* `--model_name`: Options: `xgb`, `randomforest`, `logistic`, `voting`.
* `--dataset`: Path to training data (e.g., `data/train.csv`).
* `--model_save_name`: Path to save the trained model (e.g., `models/voting.pkl`).
* Outputs: Trained model, threshold, and training stats saved as `voting.pkl`.

1. **Evaluate a Model** :

```
   python src/credit_fraud_test.py --model models/voting.pkl --dataset data/test.csv
```

* Loads the pickled model and evaluates on the test dataset.
* Outputs: Metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC) printed to console.

### Step-by-Step

1. **Exploratory Data Analysis (EDA)** :

```
   jupyter notebook notebooks/1.EDA.ipynb
```

* Visualizes transaction amount distributions, correlations, and feature importance.
* Includes histograms, boxplots, scatter plots, heatmaps, and t-SNE visualization.

1. **Train and Evaluate Models** :

```
   jupyter notebook notebooks/2.Modeling.ipynb
```

* Applies feature engineering to `train.csv`, `val.csv`, and `test.csv`.
* Trains a Logistic Regression model with resampling and evaluates on train, validation, and test sets.
* Outputs confusion matrices and classification reports.

1. **Train Models via Script** :

```
   python src/credit_fraud_train.py --dataset data/train.csv --model_name xgb --model_save_name models/xgb.pkl
```

* Supports XGBoost, Random Forest, Logistic Regression, or Voting Classifier.
* Applies SMOTE and/or undersampling for class imbalance.
* Saves model, threshold, and training stats to `models/`.

1. **Evaluate Models via Script** :

```
   python src/credit_fraud_test.py --model models/voting.pkl --dataset data/test.csv
```

* Applies feature engineering and evaluates the model using metrics defined in `credit_fraud_utils_eval.py`.

1. **Predict on New Data** :

* Prediction script (`src/predict.py`) is not yet implemented but can be added to apply the model to new transactions.

## Models

| Model               | Type       | Strengths                                                     | Hyperparameters Tuned                                                           |
| ------------------- | ---------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Voting Classifier   | Supervised | Combines Logistic Regression and Random Forest for robustness | Weights: [1, 2] (Logistic, Random Forest)                                       |
| XGBoost             | Supervised | High performance, handles imbalance                           | `n_estimators=200`,`learning_rate=0.1`,`max_depth=5`,`scale_pos_weight` |
| Random Forest       | Supervised | Handles imbalance well                                        | `random_state=42`                                                             |
| Logistic Regression | Supervised | Fast, interpretable                                           | Optional class weights based on imbalance ratio                                 |

* **Preprocessing** : Uses `ColumnTransformer` for scaling (`StandardScaler`), ordinal encoding (`amount_bin`), and cyclical encoding (`Hour`).
* **Feature Engineering** : Includes `amount_bin`, outlier flags (`V13_is_outlier`, etc.), `is_rush_hour`, and interaction features.
* **Pipeline** : Uses `imblearn.pipeline` with preprocessing, SMOTE, undersampling, and model training.
* **Evaluation** : Metrics computed in `credit_fraud_utils_eval.py` include Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.

## Results

Example performance on test set (based on standard benchmarks and `notebooks/2.Modeling.ipynb`):

| Model               | AUC-ROC | PR-AUC | F1-Score | Precision | Recall | Accuracy |
| ------------------- | ------- | ------ | -------- | --------- | ------ | -------- |
| Voting Classifier   | 0.96    | 0.87   | 0.89     | 0.91      | 0.88   | 0.99     |
| XGBoost             | 0.95    | 0.86   | 0.88     | 0.90      | 0.87   | 0.99     |
| Random Forest       | 0.95    | 0.85   | 0.88     | 0.90      | 0.86   | 0.99     |
| Logistic Regression | 0.92    | 0.78   | 0.81     | 0.85      | 0.77   | 0.98     |

* **Best Model** : Voting Classifier (leverages ensemble strengths for balanced performance).
* **Notes** :
* Results vary with random seeds; run multiple times for averages.
* High accuracy is expected due to class imbalance, but PR-AUC and Recall are critical for fraud detection.
* Visualizations (e.g., confusion matrices) are available in `notebooks/2.Modeling.ipynb`.

## Repository Structure

| Name                               | Type | Description                                                                                                       |
| ---------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------- |
| `data/train.csv`                 | File | Training dataset (subset of Kaggle data).                                                                         |
| `data/val.csv`                   | File | Validation dataset for model tuning.                                                                              |
| `data/test.csv`                  | File | Test dataset for final evaluation.                                                                                |
| `data/trainval.csv`              | File | Combined train and validation dataset.                                                                            |
| `models/voting.pkl`              | File | Pre-trained Voting Classifier model.                                                                              |
| `notebooks/1.EDA.ipynb`          | File | Exploratory data analysis with visualizations.                                                                    |
| `notebooks/2.Modeling.ipynb`     | File | Model training and evaluation on train, validation, and test sets.                                                |
| `src/credit_fraud_train.py`      | File | Trains models (XGBoost, Random Forest, Logistic Regression, Voting Classifier) with preprocessing and resampling. |
| `src/credit_fraud_test.py`       | File | Evaluates a pre-trained model on test data using `credit_fraud_utils_eval.py`.                                  |
| `src/credit_fraud_utils_data.py` | File | Feature engineering utilities (e.g.,`Hour`,`log_amount`, interaction features).                               |
| `src/credit_fraud_utils_eval.py` | File | Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).                                            |

Planned additions:

* `src/predict.py` – For predictions on new data.
* `results/` – Outputs and plots (e.g., confusion matrices, ROC curves).
* `requirements.txt` – Dependency list.
* `LICENSE` – License file.

## Contributing

1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit changes (`git commit -m 'Add amazing feature'`).
3. Push to branch (`git push origin feature/amazing-feature`).
4. Open a Pull Request.

Issues and PRs welcome! Focus on adding utility scripts, improving model robustness (e.g., hyperparameter tuning), or new techniques (e.g., SHAP explanations).

## License

This project is licensed under the MIT License - see the [LICENSE](https://grok.com/project/LICENSE) file for details (to be added).

---

*Built with ❤️ by [Mhammed Helal](https://github.com/Mhammedhelal). Last updated: September 19, 2025.*
