import scipy.sparse as sp
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, f1_score, accuracy_score, 
                             roc_auc_score, roc_curve, auc, 
                             confusion_matrix, ConfusionMatrixDisplay)

def save_confusion_matrix(y_test, y_pred, model_name, output_path):
    class_names = ['Negative', 'Neutral', 'Positive']
    
    # Compute the matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")

def save_roc_curve(model, X_test, y_test, model_name, output_path):
    # Get Probabilities
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        print(f"{model_name} does not support predict_proba. Skipping ROC.")
        return

    # Setup Classes
    target_classes = [0, 1, 2]
    class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    y_test_bin = label_binarize(y_test, classes=target_classes)
    
    # Setup Plot
    plt.figure(figsize=(8, 6))
    
    if hasattr(model, 'classes_'):
        model_classes = model.classes_
    else:
        model_classes = np.unique(y_test)

    lines_plotted = 0
    
    for i, cls_label in enumerate(target_classes):
        if cls_label in np.unique(y_test):
            if cls_label in model_classes:
                col_idx = np.where(model_classes == cls_label)[0][0]
                prob_scores = y_prob[:, col_idx]
                
                # Calculate ROC
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], prob_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[cls_label], lw=2,
                         label=f'{class_names[cls_label]} (AUC = {roc_auc:.2f})')
                lines_plotted += 1
            else:
                print(f"Class {cls_label} found in test data but Model didn't learn it.")

    # Finalize Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    
    if lines_plotted > 0:
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, "No Classes Plotted (Check Data)", ha='center')
        
    plt.savefig(output_path)
    plt.close()
    print(f"ROC Graph saved to {output_path}")

# Load Data
print("Loading Data...")
X_train = sp.load_npz("modelling/data/train_tfidf.npz")
X_test = sp.load_npz("modelling/data/test_tfidf.npz")

y_train_raw = pd.read_csv("modelling/data/y_train.csv").values.ravel()
y_test_raw = pd.read_csv("modelling/data/y_test.csv").values.ravel()

# Ensure Labels are Integers
le_path = "modelling/features/label_encoder.pkl"
if os.path.exists(le_path):
    le = joblib.load(le_path)
    if isinstance(y_train_raw[0], str):
        y_train = le.transform(y_train_raw)
        y_test = le.transform(y_test_raw)
    else:
        y_train = y_train_raw.astype(int)
        y_test = y_test_raw.astype(int)
else:
    y_train = y_train_raw.astype(int)
    y_test = y_test_raw.astype(int)

# TRAIN BASELINE (Calibrated)
print("\n===== Training Baseline Linear SVM (Calibrated) =====")

base_svc = LinearSVC(
    C=1.0,
    class_weight=None,
    random_state=42,
    max_iter=2000
)

svm_baseline = CalibratedClassifierCV(base_svc, method='sigmoid', cv=5)
svm_baseline.fit(X_train, y_train)

# Evaluate
y_pred_base = svm_baseline.predict(X_test)

print("\nBaseline SVM Classification Report:")
print(classification_report(y_test, y_pred_base, digits=4))

baseline_f1 = f1_score(y_test, y_pred_base, average="macro")
baseline_acc = accuracy_score(y_test, y_pred_base)

try:
    y_prob_base = svm_baseline.predict_proba(X_test)
    baseline_auc = roc_auc_score(y_test, y_prob_base, multi_class='ovr', average='macro', labels=[0, 1, 2])
except Exception as e:
    print(f"Warning: AUC Calc failed ({e})")
    baseline_auc = 0.0

print(f"Baseline Accuracy : {baseline_acc:.4f}")
print(f"Baseline Macro-F1 : {baseline_f1:.4f}")
print(f"Baseline AUC Score: {baseline_auc:.4f}")

joblib.dump(svm_baseline, "modelling/models/svm_baseline.pkl")
save_roc_curve(svm_baseline, X_test, y_test, "SVM Baseline", "modelling/models/svm_baseline_roc.png")
save_confusion_matrix(y_test, y_pred_base, "Baseline SVM", "modelling/models/svm_baseline_cm.png")

# TRAIN TUNED MODEL
print("\n===== Hyperparameter Tuning: Linear SVM =====")

# Grid Search on raw LinearSVC
param_grid = {"C": [0.01, 0.1, 1, 3, 5, 10]}
raw_svm = LinearSVC(class_weight=None, random_state=42, max_iter=3000)

grid_search = GridSearchCV(
    estimator=raw_svm,
    param_grid=param_grid,
    scoring="f1_macro", 
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)
best_C = grid_search.best_params_['C']

# Re-train BEST model with Calibration
print("\n===== Calibrating Best Tuned SVM =====")
best_raw_svm = LinearSVC(C=best_C, class_weight=None, random_state=42, max_iter=3000)
svm_tuned = CalibratedClassifierCV(best_raw_svm, method='sigmoid', cv=5)
svm_tuned.fit(X_train, y_train)

# Evaluate
y_pred_tuned = svm_tuned.predict(X_test)

print("\nTuned (Calibrated) SVM Classification Report:")
print(classification_report(y_test, y_pred_tuned, digits=4))

tuned_f1 = f1_score(y_test, y_pred_tuned, average="macro")
tuned_acc = accuracy_score(y_test, y_pred_tuned)

try:
    y_prob_tuned = svm_tuned.predict_proba(X_test)
    tuned_auc = roc_auc_score(y_test, y_prob_tuned, multi_class='ovr', average='macro', labels=[0, 1, 2])
except Exception:
    tuned_auc = 0.0

print(f"Tuned Accuracy : {tuned_acc:.4f}")
print(f"Tuned Macro-F1 : {tuned_f1:.4f}")
print(f"Tuned AUC Score: {tuned_auc:.4f}")

joblib.dump(svm_tuned, "modelling/models/svm_tuned.pkl")
save_roc_curve(svm_tuned, X_test, y_test, "SVM Tuned", "modelling/models/svm_tuned_roc.png")
save_confusion_matrix(y_test, y_pred_tuned, "Tuned SVM", "modelling/models/svm_tuned_cm.png")

# SUMMARY
print("\n===== SVM Model Comparison Summary =====")
print(f"Baseline Macro-F1 : {baseline_f1:.4f}")
print(f"Baseline AUC : {baseline_auc:.4f}\n")
print(f"Tuned Macro-F1 : {tuned_f1:.4f}")
print(f"Tuned AUC    : {tuned_auc:.4f}")

if tuned_f1 > baseline_f1:
    print("Hyperparameter tuning improved performance.")
else:
    print("Hyperparameter does not improve the performance.")