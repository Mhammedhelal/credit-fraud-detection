from abc import ABC, abstractmethod
import os
import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imb_Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from credit_fraud_utils_data import *

def _build_preprocessor(feature_meta):
    cat_feat = feature_meta['cat_feat']
    cyc_feat = feature_meta['cyc_feat']
    num_feat = feature_meta['num_feat']

    preprocessor = ColumnTransformer([
        ('oe', OrdinalEncoder(), cat_feat),
        ('scaler', StandardScaler(), num_feat),
        ('cyclical', CyclicalFeatures(cols=['Hour'], periods=[24]), cyc_feat)
    ], remainder='passthrough')

    return preprocessor

def train_model(train, models_names, model_weights, use_oversample, use_undersample,
                logistic_class_weight, n_neighbors=None):
    # --- Features ---
    X = train.drop(columns=['Class','Amount'], axis=1)
    y = train['Class']

    feature_meta = {
    "cat_feat": ['amount_bin'],
    "bin_feat": [
        'V13_is_outlier','V15_is_outlier','V22_is_outlier',
        'V23_is_outlier','V24_is_outlier','V26_is_outlier',
        'is_outlier_amount','is_rush_hour'
    ],
    "cyc_feat": ['Hour'],
    "v_feat": [f"V{i}" for i in range(1,29)]
    }
    
    used_features = (
    feature_meta["cat_feat"]
    + feature_meta["bin_feat"]
    + feature_meta["cyc_feat"]
    + feature_meta["v_feat"]
    )

    feature_meta["num_feat"] = [c for c in X.columns if c not in used_features]

    preprocessor = _build_preprocessor(feature_meta)

    # --- Sampling ---
    steps = [('preprocessor', preprocessor)]
    
    if use_oversample:
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=1)
        steps.append(('over', oversample))
        
    if use_undersample:
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1)
        steps.append(('under', undersample))

    # --- Choose Model ---
    models = []
    for model_name in models_names:
        # Get name of the model and see if it is a name of a model class
        if model_name == 'xgb':
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        elif model_name == 'randomforest':
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_name == 'logistic':
            if logistic_class_weight:
                cnter = Counter(y)
                ma = max(cnter[1], cnter[0])
                mi = min(cnter[1], cnter[0])
                ir = mi / ma
                model = LogisticRegression(class_weight={0: ir, 1: 1}, random_state=42)
            else:
                model = LogisticRegression(random_state=42)
        else:
            continue
        
        models.append((model_name, model))

    # Add model to pipeline
    if len(models) == 1:
        # Single model - no voting classifier
        steps.append(('model', models[0][1]))
    else:
        if model_weights is not None:
            if len(model_weights) != len(models):
                raise ValueError(
                    f"model_weights length {len(model_weights)} "
                    f"does not match number of models {len(models)}"
                )
        else:
            model_weights = [1] * len(models)
        # Multiple models - use voting classifier
        voting_model = VotingClassifier(
            estimators=models,
            voting='soft',
            weights=model_weights
        )
        steps.append(('model', voting_model))
    
    # --- Pipeline ---
    pipeline = imb_Pipeline(steps=steps)

    # --- Fit ---
    fitted_model = pipeline.fit(X, y)
    
    # --- Check class balance after resampling ---
    X_transformed = preprocessor.fit_transform(X)
    
    if use_oversample and not use_undersample:
        X_res, y_res = pipeline.named_steps['over'].fit_resample(X_transformed, y)
        print("After Oversampling:", Counter(y_res))

    elif use_undersample and not use_oversample:
        X_res, y_res = pipeline.named_steps['under'].fit_resample(X_transformed, y)
        print("After Undersampling:", Counter(y_res))

    elif use_oversample and use_undersample:
        X_over, y_over = pipeline.named_steps['over'].fit_resample(X_transformed, y)
        X_res, y_res = pipeline.named_steps['under'].fit_resample(X_over, y_over)
        print("After Over+Under Sampling:", Counter(y_res))
    else:
        print("Original class distribution:", Counter(y))

    return fitted_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='credit_fraud_train')
    parser.add_argument('--dataset', type=str, default='data/train.csv')
    parser.add_argument('--model_names', nargs='+', type=str, default=['logistic'], help='options: xgb, randomforest, logistic, knn')
    parser.add_argument(
    '--model_weights',
    nargs='+',
    type=float,
    default=None,
    help='weights for each model in same order as model_names'
)
    parser.add_argument('--model_save_name', type=str, default='models/model.pkl')
    parser.add_argument('--use_oversample', type=bool, default=True)
    parser.add_argument('--use_undersample', type=bool, default=True)
    parser.add_argument('--logistic_class_weight', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--n_neighbors', type=int, default=5)
    args = parser.parse_args()

    # Load data
    train = pd.read_csv(args.dataset)
  
    # Feature engineering
    train, train_stats = apply_feature_engineering(train)

    # Train model
    model = train_model(train, models_names=args.model_names, model_weights=args.model_weights, use_oversample=args.use_oversample, 
                        use_undersample=args.use_undersample, logistic_class_weight=args.logistic_class_weight, n_neighbors=args.n_neighbors)
    
    model_dict = {
        'model': model,
        'threshold': args.threshold,
        'train_stats': train_stats,
        'model_names': args.model_names
    }
    save_path = args.model_save_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_dict, save_path)