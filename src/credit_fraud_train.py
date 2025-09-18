import os
import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as imb_Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from credit_fraud_utils_data import *


def train_model(train, model_name, use_oversample, use_undersample, logistic_class_weight):
    # --- Features ---
    X = train.drop(columns=['Class','Amount'], axis=1)
    y = train['Class']

    cat_feat = ['amount_bin']   
    bin_feat = [
        'V13_is_outlier','V15_is_outlier','V22_is_outlier',
        'V23_is_outlier','V24_is_outlier','V26_is_outlier',
        'is_outlier_amount','is_rush_hour'
    ] 
    cyc_feat = ['Hour']
    v_feat = [f"V{i}" for i in range(1,29)]
    num_feat = list(X.drop(columns=cat_feat+bin_feat+cyc_feat+v_feat, axis=1).columns)

    # --- Preprocessor ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    preprocessor = ColumnTransformer([
        ('oe', OrdinalEncoder(), cat_feat),
        ('scaler', StandardScaler(), num_feat),
        ('cyclical', CyclicalFeatures(cols=['Hour'], periods=[24]), cyc_feat)
    ], remainder='passthrough')

    # --- Sampling ---
    steps = [('preprocessor', preprocessor)]
    
    if use_oversample:
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=1)
        steps.append(('over', oversample))
        
    if use_undersample:
        undersample = RandomUnderSampler(sampling_strategy='majority', random_state=1)
        steps.append(('under', undersample))

    # --- Choose Model ---
    if model_name == 'xgb':
        model = XGBClassifier(random_state=42, eval_metric='logloss')
    elif model_name == 'randomforest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'logistic':
        if logistic_class_weight:
            cnter = Counter(y)
            ma = max(cnter[1], cnter[0])
            mi = min(cnter[1], cnter[0])
            ir = mi / ma
            model = LogisticRegression(class_weight={0: ir, 1: 1}, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
    elif model_name == 'voting':
        if logistic_class_weight:
            cnter = Counter(y)
            ma = max(cnter[1], cnter[0])
            mi = min(cnter[1], cnter[0])
            ir = mi / ma
            log_reg = LogisticRegression(class_weight={0: ir, 1: 1}, random_state=42)
        else:
            log_reg = LogisticRegression(random_state=42)
            
        rf = RandomForestClassifier(random_state=42)
        model = VotingClassifier(
            estimators=[('lr', log_reg), ('rf', rf)],
            voting='soft',  # soft = average of predicted probabilities
            weights=[1,2]
        )

    # Add model to pipeline
    steps.append(('model', model))
    
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
    parser.add_argument('--model_name',type=str,default='voting', help='options: xgb, randomforest, logistic, voting')
    parser.add_argument('--model_save_name', type=str, default='models/voting.pkl')
    parser.add_argument('--use_oversample',type=bool,default=True)
    parser.add_argument('--use_undersample',type=bool,default=True)
    parser.add_argument('--logistic_class_weight',type=bool,default=False)
    parser.add_argument('--threshold',type=float,default=0.5)
    args = parser.parse_args()


    # Load data
    train = pd.read_csv(args.dataset)
  
    # Feature engineering
    train, train_stats = apply_feature_engineering(train)

    # Train model
    model = train_model(train, model_name=args.model_name, use_oversample=args.use_oversample, 
                        use_undersample=args.use_undersample, logistic_class_weight=args.logistic_class_weight)
    
    model_dict = {
        'model': model,
        'threshold': args.threshold,
        'train_stats': train_stats,
        'model_name': args.model_name
    }
    save_path = args.model_save_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_dict, save_path)

