import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, classification_report, f1_score, 
                             roc_auc_score, roc_curve, auc, 
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModel

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_confusion_matrix(y_true, y_pred, model_name, output_path):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Negative', 'Neutral', 'Positive']
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")

def save_roc_curve(y_true, y_probs, model_name, output_path):
    n_classes = 3
    y_test_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    present_classes = np.unique(y_true)
    
    for i in range(n_classes):
        if i in present_classes:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            roc_auc[i] = 0.0

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    class_names = ['Negative', 'Neutral', 'Positive']
    
    for i, color in zip(range(n_classes), colors):
        if i in roc_auc and roc_auc[i] > 0:
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC Graph saved to {output_path}")

# --- 1. BERT DATASET CLASS ---
class BertReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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

# --- 2. HYBRID ARCHITECTURES ---
class BertHybridBaseline(nn.Module):
    def __init__(self, num_classes, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        embed_dim = 768 

        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state.permute(0, 2, 1) 
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1) 
        _, (h_n, _) = self.lstm(x)
        hidden_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

class BertHybridOptimized(nn.Module):
    def __init__(self, num_classes, num_filters, kernel_size, lstm_hidden, dropout, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        embed_dim = 768 

        self.conv = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=num_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.lstm = nn.LSTM(
            input_size=num_filters, 
            hidden_size=lstm_hidden, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        hidden_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# --- 3. TRAINING ENGINE ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    try:
        if len(np.unique(all_labels)) >= 2:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])
        else:
            auc_score = 0.0
    except:
        auc_score = 0.0

    return acc, f1, auc_score, np.array(all_labels), np.array(all_probs)

# --- 4. OPTUNA OBJECTIVE ---
def objective(trial, train_dataset, test_dataset):
    params = {
        'num_filters': trial.suggest_int('num_filters', 32, 128),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
        'lstm_hidden': trial.suggest_int('lstm_hidden', 32, 128),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32])
    }

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model = BertHybridOptimized(
        num_classes=3,
        num_filters=params['num_filters'], 
        kernel_size=params['kernel_size'],
        lstm_hidden=params['lstm_hidden'],
        dropout=params['dropout'],
        freeze_bert=True 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3): 
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _, _, _ = evaluate(model, test_loader, device)
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    train_df = pd.read_csv("modelling/data/train_dl_encoded.csv")
    test_df = pd.read_csv("modelling/data/test_dl_encoded.csv")
    
    print("Loading BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = BertReviewDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=128)
    test_dataset = BertReviewDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len=128)

    # A. TRAIN BASELINE MODEL
    print("\n--- Training Baseline Hybrid Model (BERT-CNN-LSTM) ---")
    base_loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    base_loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

    baseline_model = BertHybridBaseline(num_classes=3, freeze_bert=True).to(device)
    base_optim = torch.optim.Adam(baseline_model.parameters(), lr=1e-4)
    base_crit = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train_epoch(baseline_model, base_loader_train, base_crit, base_optim, device)
        acc, f1, auc_s, _, _ = evaluate(baseline_model, base_loader_test, device)
        print(f"Baseline Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.4f}, AUC={auc_s:.4f}")

    torch.save(baseline_model.state_dict(), "modelling/models/baseline_hybrid_bert.pt")
    
    # Final Evaluation Baseline
    base_acc, base_f1, base_auc, base_y, base_probs = evaluate(baseline_model, base_loader_test, device)
    save_roc_curve(base_y, base_probs, "Baseline Hybrid BERT", "modelling/models/baseline_hybrid_roc.png")

    base_preds = np.argmax(base_probs, axis=1)
    save_confusion_matrix(base_y, base_preds, "Baseline Hybrid BERT", "modelling/models/baseline_hybrid_cm.png")

    # B. RUN OPTUNA OPTIMIZATION
    print("\n--- Running Optuna Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_dataset, test_dataset), n_trials=5)
    
    print("\nBest Hyperparameters found:")
    print(study.best_params)

    # C. TRAIN FINAL OPTIMIZED MODEL
    print("\n--- Training Final Optimized Hybrid Model ---")
    best = study.best_params
    
    final_train_loader = DataLoader(train_dataset, batch_size=best['batch_size'], shuffle=True)
    final_test_loader = DataLoader(test_dataset, batch_size=best['batch_size'], shuffle=False)

    final_model = BertHybridOptimized(
        num_classes=3,
        num_filters=best['num_filters'], 
        kernel_size=best['kernel_size'],
        lstm_hidden=best['lstm_hidden'],
        dropout=best['dropout'],
        freeze_bert=True
    ).to(device)
    
    final_optim = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
    final_crit = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train_epoch(final_model, final_train_loader, final_crit, final_optim, device)
        acc, f1, auc_s, _, _ = evaluate(final_model, final_test_loader, device)
        print(f"Optimized Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.4f}, AUC={auc_s:.4f}")

    torch.save(final_model.state_dict(), "modelling/models/best_hybrid_bert.pt")
    
    # Final Eval Optimized
    final_acc, final_f1, final_auc, final_y, final_probs = evaluate(final_model, final_test_loader, device)
    save_roc_curve(final_y, final_probs, "Optimized Hybrid BERT", "modelling/models/best_hybrid_roc.png")

    final_preds = np.argmax(final_probs, axis=1)
    save_confusion_matrix(final_y, final_preds, "Optimized Hybrid BERT", "modelling/models/best_hybrid_cm.png")

    # D. COMPARISON REPORT
    print("\n==================================")
    print(f"BASELINE: F1={base_f1:.4f} | AUC={base_auc:.4f}")
    print(f"TUNED   : F1={final_f1:.4f} | AUC={final_auc:.4f}")
    print("==================================")
    
    if final_f1 > base_f1:
        print("Hyperparameter tuning improved model performance (F1-score).")
    else:
        print("Hyperparameter tuning did not improve performance.")
        
    y_pred_final = np.argmax(final_probs, axis=1)
    print("\nClassification Report (Optimized Hybrid):")
    print(classification_report(final_y, y_pred_final, digits=4))