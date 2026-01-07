import sys
import os
import torch
import joblib
import json
import pandas as pd
import shutil
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

# SETUP PATHS
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)              
sys.path.append(parent_dir)

from backend.model_architectures import BertMultiScaleCNN, BertHybridOptimized

# CONFIG
DL_TEST_PATH = os.path.join(current_dir, "data", "test_dl_encoded.csv")
ML_FEATURES_PATH = os.path.join(current_dir, "data", "test_tfidf.npz")
ML_LABELS_PATH = os.path.join(current_dir, "data", "y_test.csv")
MODELS_DIR = os.path.join(current_dir, "models") 
FEATURES_DIR = os.path.join(current_dir, "features")
LABEL_ENCODER_PATH = os.path.join(current_dir, "features", "label_encoder.pkl")

DEPLOY_DIR = os.path.join(parent_dir, "backend", "models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET CLASS
class BertReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if pd.notna(self.texts[idx]) else ""
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# EVALUATION FUNCTIONS 
def evaluate_pytorch_bert(model, loader):
    model.eval()
    all_preds = []
    all_probs = []  
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Get Probabilities (Softmax)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    auc = 0.0
    try:
        if len(np.unique(all_labels)) >= 2:
            auc = roc_auc_score(
                all_labels, 
                all_probs, 
                multi_class='ovr', 
                average='macro', 
                labels=[0, 1, 2] 
            )
    except Exception as e:
        print(f"   [Warning] AUC Calc failed: {e}")
        
    return acc, f1, auc

def evaluate_sklearn(model, X_features, y_true, le):
    # 1. Get Predictions
    preds = model.predict(X_features)
    
    # 2. Get Probabilities (if available)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_features)
    else:
        probs = None

    # Ensure preds are Integers
    if isinstance(preds[0], str):
        try:
            preds = le.transform(preds)
        except Exception as e:
            print(f"   [Error] Could not convert string predictions to int: {e}")
            return 0, 0, 0

    y_true = y_true.astype(int)

    # 3. Calculate Metrics
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    
    auc = 0.0
    if probs is not None:
        try:
            if len(np.unique(y_true)) >= 2:
                auc = roc_auc_score(
                    y_true, 
                    probs, 
                    multi_class='ovr', 
                    average='macro',
                    labels=[0, 1, 2] 
                )
        except Exception:
            auc = 0.0
            
    return acc, f1, auc

def main():
    results = {}
    
    # LOAD LABEL ENCODER 
    print("Loading Label Encoder...")
    if os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
        print(f"   -> Classes: {le.classes_}")
    else:
        print("Error: Label Encoder not found at features/label_encoder.pkl")
        return

    # PHASE A: EVALUATE SCIKIT-LEARN MODELS
    print("\n--- Testing Scikit-Learn Models ---")
    
    if os.path.exists(ML_FEATURES_PATH) and os.path.exists(ML_LABELS_PATH):
        X_test_ml = sp.load_npz(ML_FEATURES_PATH)
        y_test_raw = pd.read_csv(ML_LABELS_PATH).values.ravel()
        
        try:
            y_test_ml = le.transform(y_test_raw)
        except:
            if isinstance(y_test_raw[0], (int, np.integer)):
                y_test_ml = y_test_raw
            else:
                 y_test_ml = y_test_raw

        print(f"   -> Loaded {X_test_ml.shape[0]} samples (TF-IDF).")
        
        sklearn_models = ['nb_baseline.pkl', 'nb_tuned.pkl', 'svm_baseline.pkl', 'svm_tuned.pkl']
        
        for model_name in sklearn_models:
            path = os.path.join(MODELS_DIR, model_name)
            if os.path.exists(path):
                print(f"Testing {model_name}...")
                try:
                    model = joblib.load(path)
                    acc, f1, auc = evaluate_sklearn(model, X_test_ml, y_test_ml, le)
                    
                    results[model_name] = {
                        "accuracy": acc,
                        "f1": f1,
                        "auc": auc, 
                        "type": "sklearn", 
                        "path": path,
                        "params": {} 
                    }
                    print(f"   -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
                except Exception as e:
                    print(f"   -> Failed: {e}")
            else:
                print(f"   -> Skipping {model_name} (Not found)")
    else:
        print("ML Test Data not found.")

    # PHASE B: EVALUATE BERT MODELS
    print("\n--- Testing PyTorch BERT Models ---")
    
    if os.path.exists(DL_TEST_PATH):
        test_df_dl = pd.read_csv(DL_TEST_PATH)
        test_df_dl['text'] = test_df_dl['text'].fillna("").astype(str)
        
        X_text_dl = test_df_dl['text'].tolist()
        y_true_dl = test_df_dl['label'].tolist()
        
        print(f"   -> Loaded {len(X_text_dl)} samples (Raw Text).")
        
        print("Loading BERT Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        test_dataset = BertReviewDataset(X_text_dl, y_true_dl, tokenizer, 128)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 1. BERT-CNN
        cnn_path = os.path.join(MODELS_DIR, "best_bert_cnn.pt")
        if os.path.exists(cnn_path):
            print("Testing BERT-CNN...") 
            cnn_params = {
                "num_classes": 3,
                "num_filters": 33,
                "kernel_sizes": [3, 4, 5],
                "dropout": 0.4279953843650618, 
                "freeze_bert": True
            }
            try:
                cnn = BertMultiScaleCNN(**cnn_params).to(device)
                cnn.load_state_dict(torch.load(cnn_path))
                acc, f1, auc = evaluate_pytorch_bert(cnn, test_loader)
                
                results['bert_cnn'] = {
                    "accuracy": acc,
                    "f1": f1, 
                    "auc": auc,
                    "type": "bert_pytorch", 
                    "class": "BertMultiScaleCNN", 
                    "params": cnn_params,
                    "path": cnn_path
                }
                print(f"   -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            except Exception as e:
                print(f"   -> Failed: {e}")

        # 2. BERT-Hybrid
        hybrid_path = os.path.join(MODELS_DIR, "best_hybrid_bert.pt")
        if os.path.exists(hybrid_path):
            print("Testing BERT-Hybrid...")
            hybrid_params = {
                "num_classes": 3,
                "num_filters": 102,  
                "kernel_size": 5,    
                "lstm_hidden": 58,   
                "dropout": 0.3209813857943808,     
                "freeze_bert": True
            }
            try:
                hybrid = BertHybridOptimized(**hybrid_params).to(device)
                hybrid.load_state_dict(torch.load(hybrid_path))
                acc, f1, auc = evaluate_pytorch_bert(hybrid, test_loader)

                results['bert_hybrid'] = {
                    "accuracy": acc,
                    "f1": f1, 
                    "auc": auc,
                    "type": "bert_pytorch", 
                    "class": "BertHybridOptimized",
                    "params": hybrid_params,
                    "path": hybrid_path
                }
                print(f"   -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            except Exception as e:
                print(f"   -> Failed: {e}")

    # PHASE C: SELECT WINNER & DEPLOY (MODIFIED)
    if not results:
        print("No models found to evaluate.")
        return

    print("\nSelection Criteria: Max Average(AUC, F1)")
    
    best_score = -1.0
    winner_name = None

    # Calculate Combined Score for each model
    print(f"\n{'Model':<20} | {'AUC':<8} | {'F1':<8} | {'Combined':<8}")
    print("-" * 50)
    
    for name, metrics in results.items():
        auc = metrics.get('auc', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        # Average of AUC and F1
        combined_score = (auc + f1) / 2
        metrics['combined_score'] = combined_score
        
        print(f"{name:<20} | {auc:.4f}   | {f1:.4f}   | {combined_score:.4f}")
        
        if combined_score > best_score:
            best_score = combined_score
            winner_name = name

    winner = results[winner_name]
    
    print(f"\n Final Model: {winner_name}")
    print(f"   Combined Score: {winner['combined_score']:.4f}")
    print(f"   AUC Score:      {winner['auc']:.4f}")
    print(f"   F1 Score:       {winner['f1']:.4f}")
    
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    
    # Copy Model
    dest_path = os.path.join(DEPLOY_DIR, "deployed_model.bin")
    shutil.copy(winner['path'], dest_path)
    
    # Copy Label Encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        shutil.copy(LABEL_ENCODER_PATH, os.path.join(DEPLOY_DIR, "label_encoder.pkl"))

    # Copy Vectorizer (only for sklearn)
    if winner['type'] == 'sklearn':
        vec_path = os.path.join(FEATURES_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            shutil.copy(vec_path, os.path.join(DEPLOY_DIR, "tfidf_vectorizer.pkl"))

    # Config
    config = {
        "model_type": winner['type'],
        "original_name": winner_name,
        "f1_score": winner['f1'],
        "accuracy": winner['accuracy'],
        "auc_score": winner['auc'],
        "combined_score": winner['combined_score']
    }
    
    if winner['type'] == 'bert_pytorch':
        config['architecture'] = winner['class']
        config['params'] = winner['params']
    
    with open(os.path.join(DEPLOY_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Deployed to {DEPLOY_DIR}")

if __name__ == "__main__":
    main()