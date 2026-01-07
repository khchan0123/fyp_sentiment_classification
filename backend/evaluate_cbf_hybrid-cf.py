import pandas as pd
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

TEST_SAMPLES = 200  
K_VALUES = [5, 10, 20, 30, 50]  
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(os.path.dirname(current_dir), "backend")
sys.path.append(backend_dir)

from recommender import HybridRecommender
from sentiment import SentimentAnalyzer

def setup_data():
    data_path = os.path.join(backend_dir, "data", "amazonindia.csv")
    try:
        df = pd.read_csv(data_path)
        df['product_id'] = df['product_id'].astype(str).str.strip()
        df['user_id'] = df['user_id'].astype(str).str.strip()
        
        analyzer = SentimentAnalyzer()
        if 'sentiment_score' not in df.columns:
            print("   -> Pre-calculating sentiments...")
            df['sentiment_score'] = df['review_content'].apply(lambda x: analyzer.get_sentiment(str(x))[0])

        # Aggregate stats
        product_stats = df.groupby('product_id').agg({
            'sentiment_score': 'mean',
            'rating': 'count'
        }).rename(columns={'sentiment_score': 'avg_sentiment'}).reset_index()
        
        unique_products = df.drop_duplicates('product_id').copy()
        products_df = pd.merge(unique_products, product_stats, on='product_id', how='left')
        products_df['avg_sentiment'] = products_df['avg_sentiment'].fillna(0)
        
        return df, products_df
    except FileNotFoundError:
        print("[ERROR] Data not found.")
        return None, None

def plot_results(results_df):
    plt.figure(figsize=(10, 6))
    
    plt.plot(
        results_df['K'], results_df['Hybrid_HitRate'], 
        marker='o', linestyle='-', linewidth=2, color='#2ca02c', label='Hybrid (Proposed)'
    )
    
    plt.plot(
        results_df['K'], results_df['PureCF_HitRate'], 
        marker='s', linestyle='--', linewidth=2, color='#d62728', label='Pure SVD++ (Baseline)'
    )
    
    plt.title('Ablation Study: Hit Rate @ K', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Recommendations (K)', fontsize=12)
    plt.ylabel('Hit Rate (Recall)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    chart_path = os.path.join(OUTPUT_DIR, "ablation_chart_3.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved to: {chart_path}")
    plt.close()

def main():
    print("Starting Recommendation System Ablation Study...")

    df, products_df = setup_data()
    if df is None: return

    print("\nInitializing Hybrid Recommender (SVD++)...")
    rec = HybridRecommender(df, products_df)

    # Generate Test Pairs
    print("\nGenerating Test Pairs...")
    user_purchases = df.groupby('user_id')['product_id'].apply(list)
    valid_users = user_purchases[user_purchases.apply(len) >= 2]
    
    test_pairs = []
    for user, purchases in valid_users.items():
        if len(purchases) >= 2:
            a, b = random.sample(purchases, 2)
            test_pairs.append((a, b))
    
    if len(test_pairs) > TEST_SAMPLES:
        test_pairs = random.sample(test_pairs, TEST_SAMPLES)
    
    print(f"   -> Testing on {len(test_pairs)} user scenarios.")

    # Run Sensitivity Analysis
    w_pure = {'w_content': 0.0, 'w_collab': 1.0}
    w_hybrid = {'w_content': 0.5, 'w_collab': 0.5}
    
    final_results = []
    
    print("\nRunning Simulations...")
    
    max_k = max(K_VALUES)
    
    hits_pure_at_k = {k: 0 for k in K_VALUES}
    hits_hybrid_at_k = {k: 0 for k in K_VALUES}
    
    for anchor, target in tqdm(test_pairs):
        # Get recommendations up to MAX K
        recs_pure = rec.get_adaptive_feed(last_interacted_id=anchor, top_n=max_k, weights=w_pure)
        recs_hybrid = rec.get_adaptive_feed(last_interacted_id=anchor, top_n=max_k, weights=w_hybrid)
        
        pure_ids = [item['product_id'] for item in recs_pure]
        hybrid_ids = [item['product_id'] for item in recs_hybrid]
        
        # Check for Hit at each K level
        for k in K_VALUES:
            if target in pure_ids[:k]:
                hits_pure_at_k[k] += 1
                
            if target in hybrid_ids[:k]:
                hits_hybrid_at_k[k] += 1

    for k in K_VALUES:
        hr_pure = hits_pure_at_k[k] / len(test_pairs)
        hr_hybrid = hits_hybrid_at_k[k] / len(test_pairs)
        
        final_results.append({
            "K": k,
            "PureCF_HitRate": hr_pure,
            "Hybrid_HitRate": hr_hybrid,
            "Improvement": hr_hybrid - hr_pure
        })

    results_df = pd.DataFrame(final_results)
    
    print("\n" + "="*50)
    print("       FINAL ABLATION RESULTS       ")
    print("="*50)
    print(results_df.to_string(index=False, formatters={
        'PureCF_HitRate': '{:.2%}'.format,
        'Hybrid_HitRate': '{:.2%}'.format,
        'Improvement': '{:+.2%}'.format
    }))
    
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results_3.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[EXPORT] Data saved to: {csv_path}")
    
    plot_results(results_df)

if __name__ == "__main__":
    main()