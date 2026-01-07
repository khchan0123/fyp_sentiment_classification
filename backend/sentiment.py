import torch
import os
import json
import joblib
import pickle
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

# Import all architectures to ensure availability
from model_architectures import (
    BertMultiScaleCNN, BertHybridOptimized
)

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.config = None
        self.model_type = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 128 

        # Setup Paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        
        self.model_path = os.path.join(models_dir, 'deployed_model.bin')
        config_path = os.path.join(models_dir, 'model_config.json')
        vec_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        le_path = os.path.join(models_dir, 'label_encoder.pkl')

        # Load Config
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.model_type = self.config.get('model_type', 'unknown')
            print(f"Config loaded. Model Type: {self.model_type}")
        except FileNotFoundError:
            print("[ERROR] model_config.json not found. Backend cannot start.")
            return

        # Load Label Encoder
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
        else:
            print("Label Encoder not found. Using default mapping.")

        # Load Model Components
        # A. SCIKIT-LEARN (SVM / NB)
        if self.model_type == 'sklearn':
            try:
                # Load Vectorizer
                if os.path.exists(vec_path):
                    self.vectorizer = joblib.load(vec_path)
                    print("   -> TF-IDF Vectorizer loaded.")
                else:
                    print("[ERROR] TF-IDF Vectorizer missing for Sklearn model!")
                
                # Load Classifier
                with open(self.model_path, 'rb') as f:
                    self.model = joblib.load(f)
                print("   -> Sklearn Classifier loaded.")
                
            except Exception as e:
                print(f"[ERROR] Failed to load Sklearn model: {e}")

        # B. PYTORCH BERT
        elif self.model_type == 'bert_pytorch':
            try:
                # Load BERT Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                print("   -> BERT Tokenizer loaded.")

                # Initialize Architecture
                arch_name = self.config.get('architecture')
                params = self.config.get('params', {})
                
                if arch_name == 'BertMultiScaleCNN':
                    self.model = BertMultiScaleCNN(**params)
                elif arch_name == 'BertHybridOptimized':
                    self.model = BertHybridOptimized(**params)
                else:
                    raise ValueError(f"Unknown architecture: {arch_name}")
                
                # Load Weights
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=torch.device('cpu'))
                )
                self.model.eval()
                print(f"   -> {arch_name} weights loaded.")

            except Exception as e:
                print(f"[ERROR] Failed to load PyTorch BERT model: {e}")

    def get_sentiment(self, text):
        if not self.model:
            return 0.0, "Neutral", 0.0

        try:
            # A: SCIKIT-LEARN
            if self.model_type == 'sklearn':
                if not self.vectorizer:
                    return 0.0, "Error", 0.0

                # Transform Text
                text_vector = self.vectorizer.transform([text])
                
                # Predict Probability
                probs = self.model.predict_proba(text_vector)[0] 
                pred_idx = np.argmax(probs)
                
                # Calculate Score (-1 to 1)
                class_map = {c: i for i, c in enumerate(self.model.classes_)}
                neg_idx = class_map.get(0, 0)
                pos_idx = class_map.get(2, 2)
                
                score = probs[pos_idx] - probs[neg_idx]
                confidence = float(np.max(probs))

            # B: PYTORCH BERT
            elif self.model_type == 'bert_pytorch':
                # Tokenize (IDs + Mask)
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                # Inference
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    probs = F.softmax(outputs, dim=1).squeeze()
                
                # Interpret
                pred_idx = torch.argmax(probs).item()
                neg_prob = probs[0].item()
                pos_prob = probs[2].item()
                
                score = pos_prob - neg_prob
                
                confidence = float(torch.max(probs).item())

            if self.label_encoder:
                label_str = self.label_encoder.inverse_transform([pred_idx])[0]
            else:
                mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                label_str = mapping.get(pred_idx, 'Neutral')

            return score, label_str, confidence

        except Exception as e:
            print(f"Inference Error: {e}")
            return 0.0, "Neutral", 0.0
