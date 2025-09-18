import argparse
from credit_fraud_utils_eval import *
from credit_fraud_utils_data import *
import joblib



parser = argparse.ArgumentParser(description='credit_fraud_test')
parser.add_argument('--model', type=str, default='models/voting.pkl')
parser.add_argument('--dataset', type=str, default='data/test.csv')
args = parser.parse_args()

loaded_model_dict = joblib.load(args.model)
model = loaded_model_dict['model']
threshold = loaded_model_dict['threshold']
train_stats = loaded_model_dict['train_stats']
name = loaded_model_dict['model_name']

data = pd.read_csv(args.dataset)
data, _ = apply_feature_engineering(data, train_stats)
eval_model(model, data, threshold=threshold)