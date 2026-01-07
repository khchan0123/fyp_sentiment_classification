import re
import pandas as pd
import numpy as np
import sys
import os

from surprise import SVDpp, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, reviews_df, products_df):
        self.reviews_df = reviews_df
        self.products_df = products_df.set_index('product_id', drop=False)
        self.content_sim = None
        self.collab_sim = None
        self.indices = None
        self.tfidf = None
        
        self.fixed_weights = {'w_content': 0.5, 'w_collab': 0.5} 
        
        print("Initializing Recommender Matrices...")
        self._build_content_model()
        self._build_collaborative_model()
        print("[HYBRID RECOMMENDER] Recommender Ready")

    def _build_content_model(self):
        # CONTENT-BASED FILTERING (TF-IDF)
        self.products_df['content_soup'] = (
            self.products_df['category'].fillna('') + " " + 
            self.products_df['product_name'].fillna('') + " " + 
            self.products_df['about_product'].fillna('')
        )
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = self.tfidf.fit_transform(self.products_df['content_soup'])
        self.content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(range(len(self.products_df)), index=self.products_df['product_id'])

    def _build_collaborative_model(self):
        print("[CF] Starting Matrix Factorization Pipeline (SVD++)...")
        BETA = 0.7 
        
        if 'sentiment_score' in self.reviews_df.columns:
            sentiment_rating = 3 + (self.reviews_df['sentiment_score'] * 2)
            self.reviews_df['augmented_rating'] = (
                (BETA * self.reviews_df['rating']) + 
                ((1 - BETA) * sentiment_rating)
            )
            value_col = 'augmented_rating'
            print(f"[CF] Augmented Ratings Calculated (Beta={BETA}). Sentiment Integrated.")
        else:
            value_col = 'rating'
            print("[CF] Warning: Using raw ratings (Sentiment missing).")

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.reviews_df[['user_id', 'product_id', value_col]], 
            reader
        )
        
        trainset = data.build_full_trainset()
        algo = SVDpp(n_factors=50, random_state=42)
        algo.fit(trainset)
        print("[CF] SVD++ Model Trained successfully.")

        latent_item_matrix = np.zeros((len(self.products_df), 50))
        found_count = 0
        for i, prod_id in enumerate(self.products_df['product_id']):
            try:
                inner_iid = trainset.to_inner_iid(prod_id)
                latent_item_matrix[i] = algo.qi[inner_iid]
                found_count += 1
            except ValueError:
                latent_item_matrix[i] = np.zeros(50)

        print(f"[CF] Latent Factors Extracted. Coverage: {found_count}/{len(self.products_df)} products.")
        self.collab_sim = cosine_similarity(latent_item_matrix, latent_item_matrix)
        self.collab_indices = self.indices 
        print("[CF] SVD++ Similarity Matrix Built.")

    # 1. HYBRID SEARCH
    def hybrid_search(self, query, category_filter=None, top_n=20):
        query_raw = query.lower().strip()
        
        synonym_map = {
            'tv': r'\b(?:tv|televisions?|smart\s?tv)\b',
            'television': r'\b(?:tv|televisions?)\b',
            'ac': r'\b(?:ac|air\s?conditioners?)\b',
            'fridge': r'\b(?:fridge|refrigerators?)\b',
            'iron': r'\b(?:iron|irons|garment\s?steamers?|steam\s?irons?)\b', 
            'phone': r'\b(?:phones?|mobiles?|smartphones?)\b',
            'headphone': r'\b(?:headphones?|earphones?|earbuds?)\b',
            'mouse': r'\b(?:mouse|mice)\b',
            'mice': r'\b(?:mouse|mice)\b'
        }
        
        tokens = query_raw.split()
        stop_words = {'for', 'and', 'with', 'to', 'in', 'the', 'a', 'of', 'compatible'}
        valid_tokens = [t for t in tokens if t not in stop_words]
        if not valid_tokens: valid_tokens = tokens 
        token_regexes = [synonym_map.get(t, rf'\b{re.escape(t)}\b') for t in valid_tokens]
        
        s1 = self.products_df['category'].astype(str).str.replace(r'([a-z])([A-Z])', r'\1 \2', regex=True)
        s2 = s1.str.replace(r'([A-Z])([A-Z][a-z])', r'\1 \2', regex=True)
        normalized_cats = s2.str.replace(r'[&,]', ' ', regex=True).str.lower()
        leaf_categories = normalized_cats.apply(lambda x: x.split('|')[-1].strip())
        product_names = self.products_df['product_name'].astype(str).str.lower()
        
        manual_scores = pd.Series(0.0, index=self.products_df.index)
        tokens_matched_count = pd.Series(0, index=self.products_df.index)

        if category_filter and category_filter != 'All':
            clean_filter = category_filter.replace('&', ' ').replace(',', ' ').lower()
            cat_filter_match = normalized_cats.str.contains(re.escape(clean_filter), case=False)
            manual_scores[cat_filter_match] += 10.0 
            manual_scores[~cat_filter_match] -= 10.0

        for regex in token_regexes:
            match_name_mask = product_names.str.contains(regex, case=False, regex=True)
            if match_name_mask.any():
                matched_names = product_names[match_name_mask]
                def get_pos_score(text):
                    m = re.search(regex, text)
                    if not m: return 0.0
                    return 1.0 - (m.start() / len(text))
                
                pos_factors = matched_names.apply(get_pos_score)
                boosts = 1.0 + (pos_factors * 2.0)
                manual_scores[match_name_mask] += boosts
                tokens_matched_count[match_name_mask] += 1
            
            match_leaf = leaf_categories.str.contains(regex, case=False, regex=True)
            manual_scores[match_leaf] += 3.0
            match_path = normalized_cats.str.contains(regex, case=False, regex=True)
            manual_scores[match_path] += 1.0

        all_tokens_matched = (tokens_matched_count >= len(valid_tokens))
        manual_scores[all_tokens_matched] += 5.0
        
        negative_keywords = [
            'pancake', 'sandwich', 'waffle', 'waffles', 'egg', 'eggs', 'boiler', 'boilers', 'fryer', 'frying',
            'lint', 'shaver', 'shavers', 'vacuum', 'vacuums', 'spray', 'bottle', 'bottles', 
            'ironing', 'board', 'cover', 'adapter', 'adapters', 'cable', 'cables', 'charger', 'chargers', 
            'connector', 'connectors', 'cord', 'cords', 'dock', 'docks', 'hub', 'hubs', 
            'powerbank', 'powerbanks', 'wire', 'wires', 'mount', 'mounts', 'stand', 'stands', 
            'bracket', 'brackets', 'pad', 'pads', 'mat', 'mats', 'case', 'cases', 'guard', 'guards', 
            'pouch', 'pouches', 'protector', 'protectors', 'screen', 'sleeve', 'sleeves', 
            'skin', 'skins', 'strap', 'straps', 'band', 'bands',
            'part', 'parts', 'replacement', 'remote', 'battery', 'batteries'
        ]
        is_negative_query = any(k in query_raw for k in negative_keywords)
        if not is_negative_query:
            pattern = '|'.join([rf'\b{k}\b' for k in negative_keywords])
            trap_match = normalized_cats.str.contains(pattern, case=False, regex=True)
            manual_scores[trap_match] -= 5.0 

        query_vec = self.tfidf.transform([query_raw])
        tfidf_matrix = self.tfidf.transform(self.products_df['content_soup'])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        final_search_scores = manual_scores.values + cosine_sim
        
        if len(final_search_scores) == 0: return []
        max_score = final_search_scores.max()
        cutoff = max(3.0, max_score * 0.25)
        if category_filter:
            cutoff = max(2.0, max_score * 0.15) 

        sorted_indices = final_search_scores.argsort()[::-1]
        valid_indices = [i for i in sorted_indices if final_search_scores[i] >= cutoff]
        
        if not valid_indices:
            print(f"Search: '{query_raw}' -> NO RESULTS")
            return []
            
        top_8_indices = valid_indices[:8]
        
        top_products_df = self.products_df.iloc[top_8_indices].copy()
        
        top_products_df['search_score'] = final_search_scores[top_8_indices]
        
        exact_matches = top_products_df.to_dict('records')
        exact_match_ids = [item['product_id'] for item in exact_matches]
        
        anchor_id = exact_match_ids[0]
        
        adaptive_recs = self.get_adaptive_feed(
            last_interacted_id=anchor_id,
            top_n=20, 
            exclude_ids=exact_match_ids, 
            lambda_param=0.5 
        )
        
        return exact_matches + adaptive_recs

    # 2. DIVERSITY (MMR)
    def _mmr_sort(self, candidates, lambda_param, top_n=10):
        # Standard Maximal Marginal Relevance (MMR)
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        selected_indices, selected_items = [], []
        remaining = candidates.copy()
        
        if remaining:
            best = remaining.pop(0)
            selected_items.append(best)
            selected_indices.append(best['index'])
            
        while len(selected_items) < top_n and remaining:
            best_mmr_score = -np.inf
            best_item_idx_in_remaining = -1
            
            for idx, candidate in enumerate(remaining):
                cand_matrix_idx = candidate['index']
                relevance = candidate['score']
                max_sim = 0.0
                
                # Check similarity with selected items
                for sel_idx in selected_indices:
                    sim = self.content_sim[cand_matrix_idx][sel_idx]
                    if sim > max_sim: max_sim = sim
                    
                # MMR Formula: Maximize Relevance, Minimize Similarity
                mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * max_sim)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item_idx_in_remaining = idx
            
            winner = remaining.pop(best_item_idx_in_remaining)
            selected_items.append(winner)
            selected_indices.append(winner['index'])
            
        return selected_items

    # 3. DISCOVERY FEED
    def get_adaptive_feed(self, last_interacted_id=None, top_n=50, exclude_ids=[], bias_category=None, lambda_param=0.5, weights=None):
        W = weights if weights else self.fixed_weights

        # Calculate Baseline Scores (Hybrid)
        if last_interacted_id and last_interacted_id in self.indices:
            idx = self.indices[last_interacted_id]
            content_scores = self.content_sim[idx]
            try:
                c_idx = self.collab_indices[last_interacted_id]
                collab_scores = self.collab_sim[c_idx]
            except:
                collab_scores = np.zeros(len(self.products_df))
        else: 
            content_scores = np.zeros(len(self.products_df))
            pop_values = np.log1p(self.products_df['rating_count'].values)
            collab_scores = (pop_values - pop_values.min()) / (pop_values.max() - pop_values.min())

        eps = 1e-9
        norm_content = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + eps)
        norm_collab = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + eps)
        
        final_scores = (W['w_content'] * norm_content) + (W['w_collab'] * norm_collab)

        # Collect Candidates & Apply Milder Bias
        candidates = []
        for i, score in enumerate(final_scores):
            prod_id = str(self.products_df.iloc[i]['product_id']).strip()
            
            if prod_id in exclude_ids: continue
            if last_interacted_id and prod_id == last_interacted_id: continue
            
            # Persona Boost (1.25x instead of aggressive 1.5x)
            if bias_category:
                cat = str(self.products_df.iloc[i]['category'])
                if bias_category.lower() in cat.lower():
                    score = score * 1.25 
            
            candidates.append({'index': i, 'score': score})

        # Apply Diversity Sort (MMR)
        diversity_pool = self._mmr_sort(candidates, lambda_param=lambda_param, top_n=top_n * 2)
        
        # Enforce Category Cap
        final_selection = []
        category_counts = {}
        max_per_category = max(3, int(top_n * 0.4)) 

        for item in diversity_pool:
            if len(final_selection) >= top_n: break
            
            idx = item['index']
            cat_raw = str(self.products_df.iloc[idx]['category'])
            cat_key = cat_raw.split('|')[-1].strip() if '|' in cat_raw else cat_raw
            
            curr_count = category_counts.get(cat_key, 0)
            if curr_count >= max_per_category:
                continue 
            
            final_selection.append(item)
            category_counts[cat_key] = curr_count + 1
            
        # Fallback
        if len(final_selection) < top_n:
            existing_indices = {x['index'] for x in final_selection}
            for item in diversity_pool:
                if len(final_selection) >= top_n: break
                if item['index'] not in existing_indices:
                    final_selection.append(item)

        # RESULTS FORMATTING
        results = []
        for item in final_selection:
            idx = item['index']
            score = item['score']
            product_data = self.products_df.iloc[idx].to_dict()
            product_data['score'] = score 
            results.append(product_data)
            
        return results

    # 4. HERO SECTION
    def get_user_recommendations(self, user_bias=None, limit=4):
        if user_bias and user_bias != "None":
            candidates = self.products_df[
                self.products_df['category'].astype(str).str.contains(user_bias, case=False)
            ].copy()
            if len(candidates) < limit: candidates = self.products_df.copy()
        else:
            candidates = self.products_df[
                self.products_df['discount_percentage'] > 30
            ].copy()

        if 'augmented_rating' in self.reviews_df.columns:
            norm_star = candidates['rating'] / 5.0
            norm_sent = ((candidates['avg_sentiment'] + 1) / 2)
            candidates['quality_score'] = (0.7 * norm_star) + (0.3 * norm_sent)
            candidates['score'] = (candidates['quality_score'] * 0.8) + \
                                  ((candidates['discount_percentage'] / 100) * 0.2)
        else:
            candidates['score'] = (candidates['rating'] / 5.0 * 0.8) + \
                                  ((candidates['discount_percentage'] / 100) * 0.2)

        return candidates.sort_values('score', ascending=False).head(limit).to_dict('records')
    
    # 5. PRODUCT DETAIL
    def get_recommendations(self, target_product_id, last_interacted_id=None, top_n=5, bias_category=None):
        return self.get_adaptive_feed(
            last_interacted_id=target_product_id, 
            top_n=top_n, 
            bias_category=bias_category
        )