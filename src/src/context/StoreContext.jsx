import React, { createContext, useContext, useState, useEffect } from 'react';
import { toast } from 'sonner';

const API_URL = "http://localhost:5000/api";

const StoreContext = createContext(null);

export const useStore = () => {
  const context = useContext(StoreContext);
  if (!context) throw new Error('useStore must be used within StoreProvider');
  return context;
};

export const StoreProvider = ({ children }) => {
  const [products, setProducts] = useState([]);
  const [userProfiles, setUserProfiles] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);
  const [recommendations, setRecommendations] = useState([]); 
  const [homeRecommendations, setHomeRecommendations] = useState([]); 
  
  // Discovery Grid
  const [smartDiscoveryFeed, setSmartDiscoveryFeed] = useState([]);
  const [isFeedLoading, setIsFeedLoading] = useState(false);
  const [isRecLoading, setIsRecLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cart, setCart] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [showSafetyModal, setShowSafetyModal] = useState(false);
  const [pendingPurchase, setPendingPurchase] = useState(null);
  const [compareList, setCompareList] = useState([]);
  
  // Search State
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchQuery, setSearchQuery] = useState(''); 

  // Helpers 
  const parsePrice = (priceStr) => {
    if (!priceStr) return 0;
    const cleanStr = String(priceStr).replace(/[₹,]/g, '').trim();
    return parseFloat(cleanStr) || 0;
  };

  const parseNumber = (numStr) => {
    if (!numStr) return 0;
    const cleanStr = String(numStr).replace(/[,]/g, '').trim();
    return parseFloat(cleanStr) || 0;
  };

  // Fetch Smart Discovery Feed
  useEffect(() => {
    const fetchDiscovery = async () => {
        if (loading) return; 
        setIsFeedLoading(true);

        try {
            const userId = currentUser ? currentUser.id : 'guest';
            const lastItem = cart.length > 0 ? cart[cart.length - 1].id : '';
            const cartIds = cart.map(item => item.id).join(',');
            
            const queryParams = new URLSearchParams({
                user_id: userId,
                last_action_id: lastItem,
                cart_ids: cartIds
            });

            const res = await fetch(`${API_URL}/discovery?${queryParams}`);
            const data = await res.json();

            const mapped = data.map(p => ({
                id: p.product_id,
                name: p.product_name,
                category: p.category ? p.category.split('|')[0] : 'General',
                price: parsePrice(p.discounted_price),
                originalPrice: parsePrice(p.actual_price),
                discount: parseNumber(p.discount_percentage),
                description: p.about_product,
                image: p.img_link,
                starRating: parseNumber(p.rating),
                ratingCount: parseNumber(p.rating_count),
                sentimentScore: p.avg_sentiment ? (p.avg_sentiment + 1) / 2 : 0.5,
                modelConfidence: p.avg_confidence !== undefined ? p.avg_confidence : 0.85,
                keywords: p.keywords || { positive: [], negative: [] }
            }));

            setSmartDiscoveryFeed(mapped);
        } catch (e) {
            console.error("Discovery Feed Error", e);
        } finally {
            setIsFeedLoading(false);
        }
    };

    const timeoutId = setTimeout(() => {
        fetchDiscovery();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [cart, currentUser, loading]); 
  
  // Fetch Home Recommendations
  useEffect(() => {
    const fetchHomeRecs = async () => {
      if (loading) return; 
      try {
        const userId = currentUser ? currentUser.id : 'guest';
        const res = await fetch(`${API_URL}/recommend?user_id=${userId}`);
        const data = await res.json();
        const mapped = data.map(p => ({
            id: p.product_id,
            name: p.product_name,
            category: p.category ? p.category.split('|')[0] : 'General',
            price: parsePrice(p.discounted_price),
            originalPrice: parsePrice(p.actual_price),
            discount: parseNumber(p.discount_percentage),
            description: p.about_product,
            image: p.img_link,
            rating: parseNumber(p.rating),
            ratingCount: parseNumber(p.rating_count),
            sentimentScore: p.avg_sentiment ? (p.avg_sentiment + 1) / 2 : 0.5,
            modelConfidence: p.avg_confidence !== undefined ? p.avg_confidence : 0.85,
            keywords: p.keywords || { positive: [], negative: [] }
        }));
        setHomeRecommendations(mapped);
      } catch (e) { console.error("Home Recs Error", e); }
    };
    fetchHomeRecs();
  }, [currentUser, loading]); 

  // Fetch Initial Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [productsRes, usersRes] = await Promise.all([
          fetch(`${API_URL}/products`),
          fetch(`${API_URL}/users`)
        ]);
        const productsData = await productsRes.json();
        const usersData = await usersRes.json();

        const mappedProducts = productsData.map(p => ({
            id: p.product_id,
            name: p.product_name,
            category: p.category ? p.category.split('|')[0] : 'General',
            fullCategory: p.category || '', 
            price: parsePrice(p.discounted_price),
            originalPrice: parsePrice(p.actual_price),
            description: p.about_product,
            image: p.img_link,
            starRating: parseNumber(p.rating),
            ratingCount: parseNumber(p.rating_count),
            sentimentScore: p.avg_sentiment ? (p.avg_sentiment + 1) / 2 : 0.5,
            keywords: p.keywords || { positive: [], negative: [] },
            sentimentBadge: p.sentiment_badge, 
            sentimentLabel: p.sentiment_label,
            modelConfidence: p.avg_confidence !== undefined ? p.avg_confidence : 0.85,
            discount: parseNumber(p.discount_percentage)
        }));

        const mappedUsers = usersData.map(u => ({
          id: u.id, name: u.name, bias: u.bias || "None", description: u.description, avatar: u.avatar
        }));

        setProducts(mappedProducts);
        setUserProfiles(mappedUsers);
        setCurrentUser(mappedUsers[0]); 
        setLoading(false);
      } catch (err) { console.error("API Error:", err); setLoading(false); }
    };
    fetchData();
  }, []);

  const fetchRecommendations = async (productId) => {
    try {
      setIsRecLoading(true);
      const userId = currentUser ? currentUser.id : 'guest';
      const lastCartItem = cart.length > 0 ? cart[cart.length - 1].id : '';
      const response = await fetch(`${API_URL}/recommend?product_id=${productId}&user_id=${userId}&last_action_id=${lastCartItem}`);
      const data = await response.json();
      const mappedRecs = data.map(p => ({
        id: p.product_id, name: p.product_name, category: p.category ? p.category.split('|')[0] : 'General',
        price: parsePrice(p.discounted_price), image: p.img_link,
        sentimentScore: p.avg_sentiment ? (p.avg_sentiment + 1) / 2 : 0.5,
        modelConfidence: p.avg_confidence !== undefined ? p.avg_confidence : 0.85,
        starRating: parseNumber(p.rating), ratingCount: parseNumber(p.rating_count),            
        keywords: p.keywords || { positive: [], negative: [] }
      }));
      setRecommendations(mappedRecs);
      setIsRecLoading(false);
    } catch (error) { setIsRecLoading(false); }
  };

  // Perform Hybrid Search
  const performSearch = async (query, category = null) => {
    setSearchQuery(query);
    
    if (!query || !query.trim()) {
        setSearchResults([]);
        return;
    }
    
    setIsSearching(true);
    try {
        let url = `${API_URL}/search?q=${encodeURIComponent(query)}`;
        
        if (category && category !== 'All') {
            url += `&category=${encodeURIComponent(category)}`;
        }
        
        const res = await fetch(url);
        const data = await res.json();
        
        const mapped = data.map(p => ({
            id: p.product_id,
            name: p.product_name,
            category: p.category ? p.category.split('|')[0] : 'General',
            price: parsePrice(p.discounted_price),
            originalPrice: parsePrice(p.actual_price),
            discount: parseNumber(p.discount_percentage),
            description: p.about_product,
            image: p.img_link,
            starRating: parseNumber(p.rating),
            ratingCount: parseNumber(p.rating_count),
            sentimentScore: p.avg_sentiment ? (p.avg_sentiment + 1) / 2 : 0.5,
            modelConfidence: p.avg_confidence !== undefined ? p.avg_confidence : 0.85,
            relevanceScore: p.search_score ? parseFloat(p.search_score.toFixed(2)) : 0,
            keywords: p.keywords || { positive: [], negative: [] }
        }));
        
        setSearchResults(mapped);
    } catch (e) {
        console.error("Search Error", e);
    } finally {
        setIsSearching(false);
    }
  };

  // Actions
  const addToCart = (product) => {
    if (product.sentimentScore < 0.4) { setPendingPurchase(product); setShowSafetyModal(true); return false; }
    setCart(prev => {
      const existing = prev.find(item => item.id === product.id);
      if (existing) { return prev.map(item => item.id === product.id ? { ...item, quantity: item.quantity + 1 } : item); }
      return [...prev, { ...product, quantity: 1 }];
    });
    return true;
  };
  const confirmPurchase = () => {
    if (pendingPurchase) {
      setCart(prev => {
        const existing = prev.find(item => item.id === pendingPurchase.id);
        if (existing) { return prev.map(item => item.id === pendingPurchase.id ? { ...item, quantity: item.quantity + 1 } : item); }
        return [...prev, { ...pendingPurchase, quantity: 1 }];
      });
    }
    setShowSafetyModal(false); setPendingPurchase(null);
  };
  const cancelPurchase = () => { setShowSafetyModal(false); setPendingPurchase(null); };
  const removeFromCart = (productId) => setCart(prev => prev.filter(item => item.id !== productId));
  const updateQuantity = (productId, quantity) => {
    if (quantity <= 0) { removeFromCart(productId); return; }
    setCart(prev => prev.map(item => item.id === productId ? { ...item, quantity } : item));
  };
  const switchUser = (userId) => { const user = userProfiles.find(u => u.id === userId); setCurrentUser(user); };
  const addToCompare = (product) => {
    if (compareList.find(p => p.id === product.id)) { toast.info("Already in comparison list"); return; }
    if (compareList.length >= 3) { toast.warning("You can only compare up to 3 products"); return; }
    setCompareList(prev => [...prev, product]); toast.success("Added to comparison");
  };
  const removeFromCompare = (productId) => setCompareList(prev => prev.filter(p => p.id !== productId));
  const clearCompare = () => setCompareList([]);
  const cartTotal = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const cartItemCount = cart.reduce((sum, item) => sum + item.quantity, 0);

  const value = {
    products, loading, error, currentUser, cart, cartTotal, cartItemCount,
    selectedProduct, showSafetyModal, pendingPurchase,
    recommendations, homeRecommendations, isRecLoading, fetchRecommendations, 
    compareList, addToCompare, removeFromCompare, clearCompare,
    smartDiscoveryFeed, isFeedLoading, 
    addToCart, confirmPurchase, cancelPurchase, removeFromCart, updateQuantity, switchUser, setSelectedProduct, userProfiles,
    searchResults, isSearching, performSearch, searchQuery 
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
};