from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def eval_model(model, data, threshold=0.5):
    X = data.drop(columns=['Class','Amount'], axis=1)
    y = data['Class']
    # --- Predict proba ---
    y_proba = model.predict_proba(X)[:, 1]

    # --- Apply threshold for predictions ---
    y_pred = (y_proba >= threshold).astype(int)

    # --- Existing metrics ---
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    # --- PR-AUC ---
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)

    # --- Print results ---
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")