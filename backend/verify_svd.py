import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import HybridRecommender
from sentiment import SentimentAnalyzer

def verify_system():
    print("Starting System Verification...")
    
    # Load Real Data
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazonindia.csv')
        reviews_df = pd.read_csv(data_path)
        reviews_df['product_id'] = reviews_df['product_id'].astype(str).str.strip()
        reviews_df['user_id'] = reviews_df['user_id'].astype(str).str.strip()
        
        print(f"   -> Loaded {len(reviews_df)} reviews.")
        
    except FileNotFoundError:
        print("Data not found at backend/data/amazonindia.csv")
        return

    # GENERATE REAL SENTIMENT SCORES
    print("\nInitializing Sentiment Engine for Verification...")
    analyzer = SentimentAnalyzer()
    
    if not analyzer.model:
        print("Sentiment Model failed to load.")
        reviews_df['sentiment_score'] = 0.0
        reviews_df['confidence'] = 0.0
    else:
        print("   -> Calculating sentiment for all reviews...")
        
        def get_sentiment_data(text):
            try:
                score, _, conf = analyzer.get_sentiment(str(text))
                return pd.Series([score, conf])
            except:
                return pd.Series([0.0, 0.0])

        tqdm.pandas(desc="Scoring Sentiments")
        reviews_df[['sentiment_score', 'confidence']] = reviews_df['review_content'].progress_apply(get_sentiment_data)
        
        print("   -> Sentiment Scoring Complete.")

    # AGGREGATE TO PRODUCT LEVEL 
    print("\n∑ Aggregating Product Metrics (Fixing KeyError)...")
    
    product_stats = reviews_df.groupby('product_id').agg({
        'sentiment_score': 'mean',
        'confidence': 'mean',
        'rating': 'count' 
    }).rename(columns={
        'sentiment_score': 'avg_sentiment',
        'confidence': 'avg_confidence',
        'rating': 'rating_count'
    }).reset_index()

    unique_products = reviews_df.drop_duplicates('product_id')[['product_id', 'product_name', 'category', 'discount_percentage', 'rating', 'about_product']]
    
    products_df = pd.merge(unique_products, product_stats, on='product_id', how='left')
    products_df['product_id'] = products_df['product_id'].astype(str).str.strip()
    products_df['avg_sentiment'] = products_df['avg_sentiment'].fillna(0)
    products_df['avg_confidence'] = products_df['avg_confidence'].fillna(0)
    
    print(f"   -> Aggregated stats for {len(products_df)} unique products.")

    # Initialize Recommender with RICH Data
    print("\nInitializing Hybrid Recommender...")
    rec = HybridRecommender(reviews_df, products_df)
    
    # INSPECT SVD MATRIX
    print("\nChecking SVD Integration...")
    
    if rec.collab_sim is None:
        print("SVD Similarity Matrix is Missing!")
        return
        
    print(f"   -> SVD Similarity Matrix Shape: {rec.collab_sim.shape}")
    
    # Check for Zeros (Sparsity)
    non_zeros = np.count_nonzero(rec.collab_sim)
    total_elements = rec.collab_sim.size
    density = (non_zeros / total_elements) * 100
    print(f"   -> Matrix Density: {density:.2f}% (Target: ~100%)")
    
    # RUN A TEST RECOMMENDATION
    print("\nRunning Test Prediction...")
    target_id = products_df.iloc[0]['product_id']
    target_name = products_df.iloc[0]['product_name']
    print(f"   -> Target Product: {target_id} ({target_name[:30]}...)")
    
    try:
        recs = rec.get_adaptive_feed(last_interacted_id=target_id, top_n=3)
        
        print(f"   -> Recommendations generated: {len(recs)}")
        for item in recs:
            print(f"      - {item['product_name'][:50]}... (Score: {item.get('score', 0):.4f})")
    except Exception as e:
        print(f"Recommendation Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_system()