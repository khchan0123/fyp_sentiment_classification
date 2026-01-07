import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from surprise import SVD, NMF, SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import KFold
from sentiment import SentimentAnalyzer

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(os.path.dirname(current_dir), "backend")
sys.path.append(backend_dir)

DATA_PATH = os.path.join(backend_dir, "data", "amazonindia.csv")
BETA_VALUES = [0.3, 0.5, 0.7] 
ALGORITHMS = {
    "SVD": SVD(random_state=42),
    "NMF": NMF(random_state=42),
    "SVD++": SVDpp(random_state=42)
}

def normalize_sentiment(score):
    return ((score + 1) * 2) + 1

def main():
    print("Starting Comprehensive Model Evaluation...")
    print(f"   Testing Betas: {BETA_VALUES}")
    print(f"   Testing Algos: {list(ALGORITHMS.keys())}")

    # 1. Load Data & Calculate Sentiment
    try:
        df = pd.read_csv(DATA_PATH)
        df['product_id'] = df['product_id'].astype(str).str.strip()
        df['user_id'] = df['user_id'].astype(str).str.strip()
    except FileNotFoundError:
        print("Data not found.")
        return

    # Check and calculate sentiment
    if 'sentiment_score' not in df.columns:
        print("\nCalculating Ground Truth Sentiments...")
        analyzer = SentimentAnalyzer()
        if not analyzer.model:
            print("Sentiment Model failed to load.")
            return
        
        def get_score(text):
            try:
                score, _, _ = analyzer.get_sentiment(str(text))
                return score
            except:
                return 0.0

        tqdm.pandas(desc="Scoring")
        df['sentiment_score'] = df['review_content'].progress_apply(get_score)
    
    df['sentiment_rating_scale'] = df['sentiment_score'].apply(normalize_sentiment)
    
    results = []
    
    kf = KFold(n_splits=5, random_state=42)
    reader = Reader(rating_scale=(1, 5))

    print("\nRunning Experiments...")
    
    for beta in BETA_VALUES:
        # Create Augmented Rating for this Beta
        col_name = f'aug_rating_b{beta}'
        df[col_name] = (beta * df['rating']) + ((1 - beta) * df['sentiment_rating_scale'])
        
        # Load into Surprise
        data = Dataset.load_from_df(df[['user_id', 'product_id', col_name]], reader)
        
        for algo_name, algo in ALGORITHMS.items():
            print(f"   -> Testing {algo_name} with Beta={beta}...")
            
            rmse_scores = []
            mae_scores = []
            nmae_scores = []
            
            # Cross Validation
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                predictions = algo.test(testset)
                
                rmse = accuracy.rmse(predictions, verbose=False)
                mae = accuracy.mae(predictions, verbose=False)
                nmae = mae / 4.0 # Normalized (5-1)
                
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                nmae_scores.append(nmae)
            
            # Record Average Metrics
            results.append({
                "Algorithm": algo_name,
                "Beta (Explicit Weight)": beta,
                "Sentiment Weight": round(1 - beta, 1),
                "RMSE": np.mean(rmse_scores),
                "MAE": np.mean(mae_scores),
                "NMAE": np.mean(nmae_scores)
            })

    # 3. Final Report
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE
    results_df = results_df.sort_values(by="RMSE", ascending=True)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS (Sorted by Performance)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    results_df.to_csv("models/model_comparison_results.csv", index=False)
    print("\nResults saved to 'modelling/model_comparison_results.csv'")

if __name__ == "__main__":
    main()