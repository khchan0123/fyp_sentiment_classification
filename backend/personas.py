import pandas as pd
from collections import Counter

# The 4 Real Users extracted from amazonindia.csv
# 1. Syed: Loves Smart TVs
# 2. Sethu: Loves USB Cables
# 3. Drishti: Loves Smartphones
# 4. Ahsan: Loves Office/Stationery
PERSONA_IDS = [
    'AE2ODWBBOBD2SITDDIEJ644OSRFQ', 
    'AEYH6IVYMLPHU62VNOKKM2KTOIIA', 
    'AEITUHHOUWUNZPQDSHA2ZWQGJUMQ', 
    'AFW6KM45ORMBEVYBQ4QMSGG2ODOQ'
]

def get_persona_profiles(reviews_df):
    profiles = []
    
    # 1. Add Default Guest
    profiles.append({
        "id": "guest",
        "name": "Guest User",
        "bias": "None",
        "description": "New visitor with no history.",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Guest"
    })

    # 2. Extract Data for Real Personas
    for uid in PERSONA_IDS:
        user_reviews = reviews_df[reviews_df['user_id'].astype(str).str.contains(uid, regex=False)]
        
        if user_reviews.empty:
            continue
            
        raw_names = user_reviews.iloc[0]['user_name'].split(',')
        name = raw_names[0] if raw_names else "Amazon Customer"

        # Determine Bias (Most Frequent Category)
        categories = []
        for cat in user_reviews['category']:
            main_cat = cat.split('|')[-1] 
            categories.append(main_cat)
            
        if not categories:
            continue
            
        top_category = Counter(categories).most_common(1)[0][0]
        
        # Create Description based on behavior
        description = f"Frequent buyer of {top_category}."
        
        profiles.append({
            "id": uid,
            "name": name,
            "bias": top_category,
            "description": description,
            "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={name}"
        })
        
    return profiles