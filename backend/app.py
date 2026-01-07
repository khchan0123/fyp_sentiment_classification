import os
import json 
try:
    import torch
except ImportError:
    pass

from google import genai
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from recommender import HybridRecommender
from sentiment import SentimentAnalyzer
from personas import get_persona_profiles

app = Flask(__name__)
CORS(app)

# Configure Gemini
GENAI_API_KEY = "AIzaSyBwsjo7t1j6RglRs6CiSDkZuhdNAufBPY0"
GEMINI_MODEL_VERSION = 'gemini-2.5-flash-lite'
GEMINI_MODEL_VERSION_2 = 'gemini-2.5-flash'
SCORING_MODEL_NAME = "CNN" 
DATASET_NAME = "Amazon Sales Data"

client = genai.Client(api_key=GENAI_API_KEY)

# GLOBAL VARIABLES 
products_df = None
reviews_df = None
recommender = None
analyzer = SentimentAnalyzer()
user_profiles_cache = []

# DATA LOADING
def load_data():
    global products_df, reviews_df, recommender, user_profiles_cache

    print("Loading Data...")
    try:
        df = pd.read_csv('data/amazonindia.csv')
        df['review_content'] = df['review_content'].astype(str)
    except FileNotFoundError:
        print("Error: amazonindia.csv not found in backend/data/")
        return

    # Data Cleaning
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
    
    # Run Sentiment Analysis
    print("Running Sentiment Analysis...")
    results = df['review_content'].apply(analyzer.get_sentiment)

    df['sentiment_score'], df['sentiment_label'], df['confidence'] = zip(*results)
    reviews_df = df

    # Create Product Aggregates
    print("Aggregating Product Data...")
    products_df = df.groupby('product_id').agg({
        'product_name': 'first',
        'category': 'first',
        'discounted_price': 'first',
        'actual_price': 'first',
        'discount_percentage': 'first',
        'rating': 'mean',
        'rating_count': 'first',
        'about_product': 'first',
        'img_link': 'first',
        'sentiment_score': 'mean',
        'confidence': 'mean'
    }).reset_index()

    products_df.rename(columns={
        'sentiment_score': 'avg_sentiment', 
        'confidence': 'avg_confidence'
    }, inplace=True)
    
    # Initialize "keywords"
    products_df['keywords'] = products_df['product_id'].apply(lambda x: {'positive': [], 'negative': []})

    # Badges
    def get_badge(row):
        if row['avg_sentiment'] > 0.2 and row['rating'] > 4.0 and row['avg_confidence'] > 0.7: 
            return 'Verified Quality'
        if row['avg_sentiment'] < 0.0 and row['rating'] > 4.0: 
            return 'Misleading Rating'
        return 'General'
    
    products_df['sentiment_badge'] = products_df.apply(get_badge, axis=1)

    # Initialize Recommender
    recommender = HybridRecommender(reviews_df, products_df)
    
    # Generate Personas
    print("Generating User Personas...")
    user_profiles_cache = get_persona_profiles(reviews_df)
    
    print(f"Server Ready. Loaded {len(products_df)} products and {len(user_profiles_cache)} personas.")

def get_reviews_for_product(product_id):
    if reviews_df is None or reviews_df.empty:
        print("Error: Reviews DataFrame is empty")
        return ""
        
    try:
        print(f"Searching for Product ID: {product_id} (Type: {type(product_id)})")
        mask = reviews_df['product_id'].astype(str).str.strip() == str(product_id).strip()
        product_reviews = reviews_df[mask]['review_content'].tolist()
        print(f"Found {len(product_reviews)} reviews for ID {product_id}")
        
        if not product_reviews:
            return ""

        return " ".join(product_reviews[:15]) 
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        return ""

# ROUTES
@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    product_id = data.get('product_id')
    product_name = data.get('product_name')

    print(f"Received Request: Analyze {product_name} ({product_id})")

    # Get real reviews
    reviews_text = get_reviews_for_product(product_id)

    # Prepare Response Structure with Metadata
    response_data = {
        "positive": [],
        "negative": [],
        "meta": {
            "summary_model": GEMINI_MODEL_VERSION,
            "scoring_model": SCORING_MODEL_NAME,
            "dataset": DATASET_NAME
        }
    }

    if not reviews_text:
        response_data["positive"] = ["No Data Available"]
        response_data["negative"] = ["No Data Available"]
        return jsonify(response_data), 200

    # Construct Prompt
    prompt = f"""
    Analyze the following customer reviews for the product '{product_name}'.
    Identify the top 3 most distinct Positive traits (Pros) and top 3 Negative traits (Cons).
    
    Return ONLY a raw JSON object with this structure:
    {{
      "positive": ["Adjective1", "Adjective2", "Adjective3"],
      "negative": ["Adjective1", "Adjective2", "Adjective3"]
    }}

    Reviews:
    {reviews_text[:3000]} 
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_VERSION,
            contents=prompt
        )
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
             response_text = response_text[3:-3]
        
        parsed_json = json.loads(response_text)
        
        response_data["positive"] = parsed_json.get("positive", [])
        response_data["negative"] = parsed_json.get("negative", [])
            
        return jsonify(response_data), 200

    except Exception as e:
        print(f"[LLM Error] Gemini API Error: {e}")
        response_data["positive"] = ["Reliable (Fallback)", "Available"]
        response_data["negative"] = ["Mixed Reviews (Fallback)"]
        return jsonify(response_data), 200

@app.route('/api/products', methods=['GET'])
def get_products():
    query = request.args.get('search', '').lower()
    if query:
        filtered = products_df[
            products_df['product_name'].str.lower().str.contains(query) | 
            products_df['category'].str.lower().str.contains(query)
        ]
    else:
        filtered = products_df
    return jsonify(filtered.to_dict(orient='records'))

@app.route('/api/users', methods=['GET'])
def get_users():
    # Returns the real personas extracted from CSV
    return jsonify(user_profiles_cache)

@app.route('/api/recommend', methods=['GET'])
def recommend():
    # Hybrid Recommendation Endpoint 
    product_id = request.args.get('product_id')
    user_id = request.args.get('user_id')
    last_action_id = request.args.get('last_action_id')
    
    bias_category = None

    # Apply Persona Bias logic
    if user_id and user_id != 'guest':
        user = next((u for u in user_profiles_cache if u['id'] == user_id), None)
        if user:
            bias_category = user['bias']
    
    # A: Home Page Recommendations (User-to-Item)
    if not product_id:
        recs = recommender.get_user_recommendations(user_bias=bias_category)
        
        for r in recs:
            if 'sentiment_std' in r and (pd.isna(r['sentiment_std'])):
                r['sentiment_std'] = 0.0
        return jsonify(recs)
    
    # B: Product Page Recommendations (Item-to-Item)
    recs = recommender.get_recommendations(
        target_product_id=product_id, 
        last_interacted_id=last_action_id,
        top_n=5,
        bias_category=bias_category 
    )

    for r in recs:
        if 'sentiment_std' in r and (pd.isna(r['sentiment_std'])):
            r['sentiment_std'] = 0.0
            
    return jsonify(recs)

@app.route('/api/discovery', methods=['GET'])
def discovery_feed():
    user_id = request.args.get('user_id')
    last_action_id = request.args.get('last_action_id')

    cart_ids_param = request.args.get('cart_ids', '')
    cart_ids = [cid.strip() for cid in cart_ids_param.split(',') if cid.strip()]

    if last_action_id:
        last_action_id = str(last_action_id).strip()

    # Identify User Persona
    bias_category = None
    if user_id and user_id != 'guest':
        user = next((u for u in user_profiles_cache if u['id'] == user_id), None)
        if user:
            bias_category = user['bias']

    # Get Hero Items (to exclude them from grid)
    hero_recs = recommender.get_user_recommendations(user_bias=bias_category, limit=4)
    hero_ids = [str(r['product_id']).strip() for r in hero_recs]

    # Combine Exclusion Lists (Hero + Cart)
    total_exclusion_list = list(set(hero_ids + cart_ids))

    # Generate Smart Feed
    feed = recommender.get_adaptive_feed(
        last_interacted_id=last_action_id,
        top_n=50,
        exclude_ids=total_exclusion_list,
        bias_category=bias_category
    )
    
    for r in feed:
        if 'sentiment_std' in r and (pd.isna(r['sentiment_std'])):
            r['sentiment_std'] = 0.0

    return jsonify(feed)

@app.route('/api/search', methods=['GET'])
def search_products():
    query = request.args.get('q', '').lower()
    category = request.args.get('category', None)
    
    if not query:
        return jsonify([])

    results = recommender.hybrid_search(query, category_filter=category)
    
    for r in results:
        if 'sentiment_std' in r and (pd.isna(r['sentiment_std'])):
            r['sentiment_std'] = 0.0
            
    return jsonify(results)

@app.route('/api/explain', methods=['POST'])
def explain_recommendations():
    data = request.json
    query = data.get('query', '')
    candidates = data.get('products', [])[:8] 
    
    if not candidates:
        return jsonify({"explanation": "I couldn't find enough data to explain these picks."})

    # Prepare Summary
    product_profiles = []
    for i, p in enumerate(candidates):
        relevance_score = p.get('relevanceScore', 0)
        sentiment_val = p.get('sentimentScore', 0.5)
        sentiment_pct = int(sentiment_val * 100)
        price = p.get('price', 0)
        rating = p.get('starRating', 0)
        
        pos_keywords = p.get('keywords', {}).get('positive', [])
        features = ', '.join(pos_keywords[:2]) if pos_keywords else "quality"
        
        profile = (
            f"Item ID {i} (Rank #{i+1}): {p.get('name', 'Item')[:60]}... "
            f"[Price: ₹{price}, Relevance Score: {relevance_score:.1f}, Sentiment: {sentiment_pct}% Positive]. "
            f"Features: {features}."
        )
        product_profiles.append(profile)
    
    products_text = "\n".join(product_profiles)
    
    # 3. PROMPT
    prompt = f"""
    You are an intelligent shopping assistant.
    
    User Query: "{query}"
    
    Here are the Top 8 Algorithm Matches (Ranked by Text Relevance):
    {products_text}
    
    --- PHASE 1: SELECTION STRATEGY (STRICT) ---
    You must select up to 3 items, but ONLY if they meet these strict quality standards:
    
    1. **The Winner:** Must be the Item with Rank #1 (Highest Relevance).
    2. **Alternative A (Quality Pick):** Scan the remaining items. Pick the one with the HIGHEST Sentiment Score. 
       - **CRITICAL:** If its Sentiment Score is below 60%, DO NOT select an Alternative A. Stop here.
    3. **Alternative B (Value/Feature Pick):** Scan for another item with a Sentiment Score > 75% that has a different price point. 
       - **CRITICAL:** If NO item has > 75% sentiment, DO NOT select an Alternative B.
       - **CRITICAL:** If the selected item has a sentiment score < 50% (like 8%), YOU ARE FORBIDDEN FROM RECOMMENDING IT.

    --- PHASE 2: PERSONA & TONE ---
    INFER THE PERSONA based on category:
    - Tech/Gaming -> Expert, spec-focused tone.
    - Home/Kitchen -> Warm, helpful, family-oriented tone.
    - Office/Tools -> Practical, no-nonsense tone.
    - Fashion/Lifestyle -> Enthusiastic, trend-focused tone.

    --- PHASE 3: GENERATE OUTPUT ---
    Write a conversational insight (3-4 sentences max):
    
    1. **The Winner:** Explain why Rank #1 is the top choice.
       - **RULE:** State that it is the most relevant match or fits your search perfectly using your words that dynamically change.
       - **FORBIDDEN:** Do NOT mention the numeric 'Relevance Score' (e.g. don't say "Score 12.5"). Users don't understand that math.
       
    2. **The Alternatives:** - If you found valid high-quality alternatives (>75% sentiment), introduce them: "However, for a higher-rated option..." or "If you want a budget pick..."
       - If NO alternatives met the quality bar, simply say: "This is by far the safest choice as other options currently lack sufficient positive feedback."
       
    3. **The Proof:** When mentioning the Alternatives, you MUST quote their "Sentiment Score" (e.g. "96% positive feedback") to prove they are better quality.
    
    Output ONLY the explanation text.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_VERSION_2,
            contents=prompt
        )
        return jsonify({"explanation": response.text.strip()})
    except Exception as e:
        print(f"Explanation Error: {e}")
        return jsonify({"explanation": f"The top result is the most relevant match for '{query}', but I've also found some highly-rated alternatives with excellent sentiment scores if you're looking for better value or quality."})
    
if __name__ == '__main__':
    load_data()
    app.run(debug=True, port=5000)