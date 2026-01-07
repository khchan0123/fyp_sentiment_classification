import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment import SentimentAnalyzer

def debug_scoring():
    print("Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    if not analyzer.model:
        print("Model failed to load. Check paths.")
        return

    # Inspect Model Classes
    print(f"\nModel Type: {analyzer.model_type}")
    if analyzer.model_type == 'sklearn':
        classes = analyzer.model.classes_
        print(f"   Raw Classes in Model: {classes}")
    else:
        print("   (PyTorch Model - Classes fixed as 0:Negative, 1:Neutral, 2:Positive)")

    # Load Real Data
    print("\nLoading Data Sample...")
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazonindia.csv')
        df = pd.read_csv(data_path)
        
        sample = df['review_content'].astype(str).sample(5, random_state=42).tolist()
        
        sample.append("This product is absolutely terrible. Worst waste of money ever.")
        sample.append("It's okay, not great but does the job.")
        sample.append("Amazing! Best purchase I've made all year.")
        
    except FileNotFoundError:
        print("Data file not found. Using synthetic examples only.")
        sample = [
            "Terrible product, do not buy.", 
            "It is average.", 
            "Excellent quality, loved it!"
        ]

    # Detailed Scoring Breakdown
    print("\nScoring Breakdown:")
    print("-" * 85)
    print(f"{'Review Snippet':<40} | {'Neg Prob':<10} | {'Pos Prob':<10} | {'Final Score':<10}")
    print("-" * 85)

    for text in sample:
        snippet = (text[:37] + '...') if len(text) > 37 else text
        
        # A. SKLEARN LOGIC
        if analyzer.model_type == 'sklearn':
            vec = analyzer.vectorizer.transform([text])
            if hasattr(analyzer.model, "predict_proba"):
                probs = analyzer.model.predict_proba(vec)[0]
            else:
                d_scores = analyzer.model.decision_function(vec)[0]
                probs = (np.exp(d_scores) / np.sum(np.exp(d_scores)))

            if len(probs) == 3:
                neg_prob = probs[0]
                pos_prob = probs[2]
            else:
                neg_prob = probs[0]
                pos_prob = probs[1]

        # B. PYTORCH BERT LOGIC
        elif analyzer.model_type == 'bert_pytorch':
            encoding = analyzer.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=analyzer.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Predict
            with torch.no_grad():
                outputs = analyzer.model(input_ids, attention_mask)
                probs = F.softmax(outputs, dim=1).squeeze().numpy()
            
            neg_prob = probs[0]
            pos_prob = probs[2]

        # CALC SCORE
        score = pos_prob - neg_prob
            
        print(f"{snippet:<40} | {neg_prob:.4f}     | {pos_prob:.4f}     | {score:.4f}")
            
    print("-" * 85)

if __name__ == "__main__":
    debug_scoring()