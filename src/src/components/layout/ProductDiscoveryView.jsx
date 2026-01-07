import React, { useState, useMemo, useEffect } from 'react';
import { useStore } from '../../context/StoreContext';
import { ProductCard } from '../product/ProductCard';
import { Star, ChevronRight, ArrowLeft, ShieldCheck, Filter, ArrowUpDown, Sparkles, Search, Check, Info, RefreshCw, Tag, Bot, X, MessageSquare } from 'lucide-react';
import { EnhancedGauge, RatingDistributionChart } from '../analytics/SentimentChart';

export const ProductDiscoverView = ({ onProductClick, onBack, onClearSearch }) => {
  const { products, searchResults, isSearching, performSearch, searchQuery } = useStore();
  
  // --- STATE ---
  const [visibleCounts, setVisibleCounts] = useState({}); 
  const [activeFilter, setActiveFilter] = useState('All'); 
  const [selectedSearchCategory, setSelectedSearchCategory] = useState('All'); 
  const [sortBy, setSortBy] = useState('relevance'); 
  
  // AI Assistant State
  const [explanation, setExplanation] = useState("");
  const [isExplaining, setIsExplaining] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false); // Controls visibility of the bubble
  
  const [filters, setFilters] = useState({
    minRating: false,    
    highSentiment: false, 
    priceRange: 'all'     
  });

  const INITIAL_COUNT = 8; 
  const LOAD_INCREMENT = 12; 

  // Reset when query changes
  useEffect(() => {
      setSelectedSearchCategory('All');
      setExplanation(""); 
      setIsChatOpen(false); // Close chat on new search
  }, [searchQuery]);

  // --- FETCH AI EXPLANATION ---
  useEffect(() => {
      if (searchResults && searchResults.length > 0 && searchQuery) {
          setIsExplaining(true);
          // 1. Open the chat "Thinking" state immediately to show responsiveness
          setIsChatOpen(true); 
          
          fetch('http://localhost:5000/api/explain', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  type: 'search',
                  query: searchQuery,
                  products: searchResults.slice(0, 3) 
              })
          })
          .then(res => res.json())
          .then(data => {
              setExplanation(data.explanation);
              // 2. Ensure chat stays open to show result
              setIsChatOpen(true);
          })
          .catch(err => {
              console.error("Failed to fetch explanation", err);
              setIsChatOpen(false); // Hide if failed
          })
          .finally(() => setIsExplaining(false));
      }
  }, [searchResults, searchQuery]);

  const availableSearchCategories = useMemo(() => {
      if (!searchResults || searchResults.length === 0) return [];
      const cats = new Set();
      searchResults.forEach(p => {
          if (p.category) {
              const root = p.category.split('|')[0].trim();
              cats.add(root);
          }
      });
      return ['All', ...Array.from(cats).sort()];
  }, [searchResults]);

  const handleSearchCategoryClick = (cat) => {
      setSelectedSearchCategory(cat);
      performSearch(searchQuery, cat);
  };

  // --- HELPER: Common Filter & Sort Logic ---
  const processList = (items) => {
      // 1. Attribute Filter (Client-Side)
      let filtered = items.filter(p => {
          if (filters.minRating && (p.starRating || 0) < 4.0) return false;
          if (filters.highSentiment && (p.sentimentScore || 0) < 0.8) return false;

          if (filters.priceRange !== 'all') {
              const price = p.price || 0;
              if (filters.priceRange === 'under500' && price >= 500) return false;
              if (filters.priceRange === '500-1000' && (price < 500 || price > 1000)) return false;
              if (filters.priceRange === '1000-5000' && (price < 1000 || price > 5000)) return false;
              if (filters.priceRange === '5000+' && price <= 5000) return false;
          }
          return true;
      });

      // 2. Sort
      if (sortBy !== 'relevance') {
          filtered.sort((a, b) => {
              if (sortBy === 'priceAsc') return a.price - b.price;
              if (sortBy === 'priceDesc') return b.price - a.price;
              if (sortBy === 'rating') return b.starRating - a.starRating;
              if (sortBy === 'sentiment') return b.sentimentScore - a.sentimentScore;
              return 0; 
          });
      }
      return filtered;
  };

  // --- 1. Process Data (Search Mode) ---
  const processedSearch = useMemo(() => {
      if (!searchResults) return { top: [], recs: [] };
      const rawTop = searchResults.slice(0, 8);
      const rawRecs = searchResults.slice(8);
      return { top: processList(rawTop), recs: processList(rawRecs) };
  }, [searchResults, filters, sortBy]);

  // --- 2. Process Data (Category Mode) ---
  const groupedData = useMemo(() => {
    if (!products.length) return {};

    const filteredProducts = products.filter(p => {
        if (filters.minRating && (p.starRating || 0) < 4.0) return false;
        if (filters.highSentiment && (p.sentimentScore || 0) < 0.8) return false;
        if (filters.priceRange !== 'all') {
            const price = p.price || 0;
            if (filters.priceRange === 'under500' && price >= 500) return false;
            if (filters.priceRange === '500-1000' && (price < 500 || price > 1000)) return false;
            if (filters.priceRange === '1000-5000' && (price < 1000 || price > 5000)) return false;
            if (filters.priceRange === '5000+' && price <= 5000) return false;
        }
        return true;
    });

    const groups = {};
    filteredProducts.forEach(p => {
      const cat = p.category || 'Uncategorized';
      if (!groups[cat]) {
        groups[cat] = {
          items: [], totalSentiment: 0, totalRating: 0, count: 0,
          ratingDist: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
        };
      }
      
      groups[cat].items.push(p);
      groups[cat].totalSentiment += p.sentimentScore || 0.5;
      groups[cat].totalRating += p.starRating || 0;
      groups[cat].count += 1;

      const r = Math.round(p.starRating || 0);
      if (r >= 1 && r <= 5) groups[cat].ratingDist[r] += 1;
    });

    Object.keys(groups).forEach(cat => {
      const group = groups[cat];
      group.avgSentiment = group.totalSentiment / group.count;
      group.avgRating = group.totalRating / group.count;

      group.items.sort((a, b) => {
          if (sortBy === 'priceAsc') return a.price - b.price;
          if (sortBy === 'priceDesc') return b.price - a.price;
          if (sortBy === 'rating') return b.starRating - a.starRating;
          if (sortBy === 'sentiment') return b.sentimentScore - a.sentimentScore;
          return 0; 
      });
    });

    return groups;
  }, [products, filters, sortBy]);

  const filteredCategories = activeFilter === 'All' ? Object.keys(groupedData) : [activeFilter];
  const getVisibleCount = (cat) => visibleCounts[cat] || INITIAL_COUNT;

  const handleShowMore = (cat) => {
      if (activeFilter === 'All') {
          setActiveFilter(cat);
          window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
          setVisibleCounts(prev => ({ ...prev, [cat]: (prev[cat] || INITIAL_COUNT) + LOAD_INCREMENT }));
      }
  };

  const toggleFilter = (key) => {
    setFilters(prev => ({ ...prev, [key]: !prev[key] }));
    setVisibleCounts({}); 
  };

  // --- COMPONENT: Filter Toolbar ---
  const FilterToolbar = () => (
      <div className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm mb-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
             <div className="flex items-center gap-2 text-gray-400 mr-2">
                <Filter className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Filters</span>
             </div>

             <div className="relative">
                 <select 
                    value={filters.priceRange}
                    onChange={(e) => setFilters(prev => ({ ...prev, priceRange: e.target.value }))}
                    className="appearance-none bg-gray-50 border border-gray-200 text-gray-700 text-sm rounded-lg pl-3 pr-8 py-2 focus:ring-2 focus:ring-[#3ABEF9] focus:border-transparent outline-none cursor-pointer hover:bg-gray-100 transition-colors"
                 >
                    <option value="all">Price: All</option>
                    <option value="under500">Under ₹500</option>
                    <option value="500-1000">₹500 - ₹1,000</option>
                    <option value="1000-5000">₹1,000 - ₹5,000</option>
                    <option value="5000+">Over ₹5,000</option>
                 </select>
                 <ChevronRight className="w-4 h-4 text-gray-400 absolute right-2 top-1/2 transform -translate-y-1/2 rotate-90 pointer-events-none" />
             </div>

             <div className="group relative flex items-center">
                 <div className="mr-2 relative group/tooltip">
                    <Info className="w-4 h-4 text-gray-400 cursor-help hover:text-[#3ABEF9] transition-colors" />
                    <div className="absolute top-1/2 right-full -translate-y-1/2 mr-2 w-48 p-2 bg-gray-800 text-white text-[10px] rounded-lg shadow-xl opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 text-center">
                        Shows items with more than 80% Positive AI Sentiment Score
                        <div className="absolute top-1/2 -right-1 -translate-y-1/2 border-8 border-transparent border-l-gray-800"></div>
                    </div>
                 </div>

                 <button
                    onClick={() => toggleFilter('highSentiment')}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border transition-all ${
                        filters.highSentiment 
                        ? 'bg-emerald-50 border-emerald-200 text-emerald-700' 
                        : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                    }`}
                 >
                    {filters.highSentiment && <Check className="w-3.5 h-3.5" />}
                    <ShieldCheck className="w-4 h-4" />
                    High Truth Score
                 </button>
             </div>

             <button
                onClick={() => toggleFilter('minRating')}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border transition-all ${
                    filters.minRating 
                    ? 'bg-yellow-50 border-yellow-200 text-yellow-700' 
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
             >
                {filters.minRating && <Check className="w-3.5 h-3.5" />}
                <Star className="w-4 h-4" />
                4★ & Above
             </button>
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto border-t md:border-t-0 border-gray-100 pt-3 md:pt-0">
             <div className="flex items-center gap-2 text-gray-400">
                <ArrowUpDown className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-wider">Sort</span>
             </div>
             <div className="relative flex-grow md:flex-grow-0">
                 <select 
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="w-full md:w-48 appearance-none bg-gray-50 border border-gray-200 text-gray-700 text-sm rounded-lg pl-3 pr-8 py-2 focus:ring-2 focus:ring-[#3ABEF9] focus:border-transparent outline-none cursor-pointer hover:bg-gray-100 transition-colors"
                 >
                    <option value="relevance">Relevance</option>
                    <option value="priceAsc">Price: Low to High</option>
                    <option value="priceDesc">Price: High to Low</option>
                    <option value="rating">Highest Rated</option>
                    <option value="sentiment">Best AI Truth Score</option>
                 </select>
                 <ChevronRight className="w-4 h-4 text-gray-400 absolute right-2 top-1/2 transform -translate-y-1/2 rotate-90 pointer-events-none" />
             </div>
          </div>
      </div>
  );

  const isSearchMode = (searchQuery && searchQuery.trim().length > 0) || isSearching;

  // FLOATING ASSISTANT COMPONENT
  const FloatingAssistant = () => (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end pointer-events-none">
        {/* Chat Bubble */}
        <div 
            className={`pointer-events-auto mb-4 mr-2 max-w-xs md:max-w-sm bg-white rounded-2xl shadow-2xl border border-gray-100 transform transition-all duration-300 origin-bottom-right ${
                isChatOpen ? 'scale-100 opacity-100 translate-y-0' : 'scale-90 opacity-0 translate-y-4 pointer-events-none'
            }`}
        >
            {/* Header */}
            <div className="bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] px-4 py-3 rounded-t-2xl flex items-center justify-between shadow-sm">
                <div className="flex items-center gap-2 text-white">
                    <Bot className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">AI Insight</span>
                </div>
                <button 
                    onClick={() => setIsChatOpen(false)}
                    className="text-white/80 hover:text-white transition-colors"
                >
                    <X className="w-4 h-4" />
                </button>
            </div>
            
            {/* Content */}
            <div className="p-5 text-sm text-gray-700 leading-relaxed relative">
                {isExplaining ? (
                    /* Loading State */
                    <div className="flex items-center gap-3 text-[#3ABEF9]">
                        <span className="relative flex h-3 w-3">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#3ABEF9] opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-3 w-3 bg-[#3ABEF9]"></span>
                        </span>
                        <span className="font-medium animate-pulse">Analyzing search intent...</span>
                    </div>
                ) : (
                    /* Explanation Text */
                    <div className="animate-fade-in font-medium">
                        {explanation || "I can help explain why these products were chosen for you."}
                    </div>
                )}
                
                <div className="absolute -bottom-2 right-4 w-4 h-4 bg-white transform rotate-45 border-b border-r border-gray-100"></div>
            </div>
        </div>

        {/* Floating Toggle Button */}
        <button 
            onClick={() => setIsChatOpen(!isChatOpen)}
            className={`pointer-events-auto group relative flex items-center justify-center w-14 h-14 rounded-full shadow-xl transition-all duration-300 ${
                isChatOpen 
                ? 'bg-gray-900 rotate-90 text-white' 
                : 'bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] hover:scale-110 text-white'
            }`}
        >
            {isChatOpen ? (
                <X className="w-6 h-6" />
            ) : (
                <>
                    <Sparkles className="w-6 h-6 animate-pulse" />
                    {!isChatOpen && explanation && (
                        <span className="absolute top-0 right-0 -mt-1 -mr-1 flex h-4 w-4">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-4 w-4 bg-red-500 border-2 border-white"></span>
                        </span>
                    )}
                </>
            )}
        </button>
    </div>
  );

  // Loading State
  if (isSearching) {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-[#F8FAFC]">
             <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-[#3ABEF9] mb-4"></div>
             <p className="text-gray-500 font-medium animate-pulse">Curating your results...</p>
        </div>
      );
  }

  // SEARCH VIEW
  if (isSearchMode) {
      const hasResultsFromBackend = searchResults && searchResults.length > 0;
      const hasFilteredResults = processedSearch.top.length > 0 || processedSearch.recs.length > 0;

      return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 min-h-screen relative">
             <button 
                onClick={() => {
                    setSelectedSearchCategory('All'); 
                    onClearSearch();
                }}
                className="flex items-center gap-2 text-gray-500 hover:text-gray-900 transition-colors mb-4 font-medium"
             >
                <ArrowLeft className="w-5 h-5" /> Back to Categories
             </button>

             <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
                 <div className="flex items-center gap-3">
                     <div className="p-3 bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] rounded-xl text-white shadow-lg">
                        <Sparkles className="w-6 h-6" />
                     </div>
                     <div>
                        <h1 className="text-3xl font-bold text-gray-900">Your Curated Results</h1>
                        <p className="text-gray-500">
                            {hasResultsFromBackend ? `${searchResults.length} matches` : '0 matches'} found for "{searchQuery}".
                        </p>
                     </div>
                 </div>
             </div>

             {/* Dynamic Category Filters */}
             {hasResultsFromBackend && availableSearchCategories.length > 1 && (
                 <div className="flex flex-wrap items-center gap-2 mb-6">
                     <span className="text-sm font-medium text-gray-500 mr-2 flex items-center gap-1">
                        <Tag className="w-4 h-4" /> Filter by:
                     </span>
                     {availableSearchCategories.map(cat => (
                         <button 
                            key={cat} 
                            onClick={() => handleSearchCategoryClick(cat)} 
                            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                                selectedSearchCategory === cat 
                                ? 'bg-gray-900 text-white shadow-md' 
                                : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
                            }`}
                         >
                            {cat}
                         </button>
                     ))}
                 </div>
             )}

             <FilterToolbar />

             {/* Results Grid Logic */}
             {!hasResultsFromBackend ? (
                 <div className="text-center py-20 bg-gray-50 rounded-3xl border border-gray-100 border-dashed animate-fade-in">
                     <div className="inline-flex p-4 bg-gray-100 rounded-full mb-4">
                        <Search className="w-8 h-8 text-gray-400" />
                     </div>
                     <h3 className="text-xl font-bold text-gray-900">No matches found</h3>
                     <p className="text-gray-500 mt-2 max-w-md mx-auto">
                        We couldn't find any items matching "{searchQuery}". Try using broader keywords.
                     </p>
                     <button onClick={() => { setSelectedSearchCategory('All'); performSearch(''); onClearSearch(); }} className="mt-6 px-6 py-2 bg-white border border-gray-200 text-gray-700 font-bold rounded-full shadow-sm hover:shadow-md transition-all inline-flex items-center gap-2">
                        <RefreshCw className="w-4 h-4" /> Clear Search
                     </button>
                 </div>
             ) : !hasFilteredResults ? (
                 <div className="text-center py-20 bg-gray-50 rounded-3xl border border-gray-100 border-dashed animate-fade-in">
                     <div className="inline-flex p-4 bg-gray-100 rounded-full mb-4">
                        <Filter className="w-8 h-8 text-gray-400" />
                     </div>
                     <h3 className="text-xl font-bold text-gray-900">No matches with current filters</h3>
                     <button onClick={() => { setFilters({ minRating: false, highSentiment: false, priceRange: 'all' }); setSelectedSearchCategory('All'); if (selectedSearchCategory !== 'All') performSearch(searchQuery, 'All'); }} className="mt-6 px-6 py-2 bg-gray-900 text-white font-bold rounded-full shadow-lg hover:shadow-xl transition-all inline-flex items-center gap-2">
                        <RefreshCw className="w-4 h-4" /> Reset Filters
                     </button>
                 </div>
             ) : (
                 <>
                     {processedSearch.top.length > 0 && (
                     <section className="mb-12 animate-fade-in">
                         <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                            <Search className="w-5 h-5 text-gray-400" /> Top Matches
                         </h2>
                         <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
                             {processedSearch.top.map(product => (
                                 <ProductCard key={product.id} product={product} onClick={() => onProductClick(product)} />
                             ))}
                         </div>
                     </section>
                     )}
                     {processedSearch.recs.length > 0 && (
                         <section className="animate-fade-in delay-100">
                             <div className="relative mb-8 rounded-3xl overflow-hidden">
                                <div className="absolute inset-0 bg-gradient-to-br from-[#3ABEF9] to-[#A7D397] opacity-25"></div>
                                <div className="relative p-8 md:p-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 border border-blue-100 rounded-3xl">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <h2 className="text-2xl font-bold text-gray-900">You Might Also Need</h2>
                                        </div>
                                        <p className="text-gray-600">Highly relevant items curated based on your search context.</p>
                                    </div>
                                    <div className="hidden md:block">
                                        <div className="px-4 py-2 bg-white rounded-full text-sm font-medium text-blue-600 shadow-sm border border-blue-100">{processedSearch.recs.length} Suggestions</div>
                                    </div>
                                </div>
                             </div>
                             <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                                 {processedSearch.recs.map(product => (
                                     <ProductCard key={product.id} product={product} onClick={() => onProductClick(product)} />
                                 ))}
                             </div>
                         </section>
                     )}
                 </>
             )}
             
             {hasResultsFromBackend && <FloatingAssistant />}
        </div>
      );
  }
  
  // DEFAULT CATEGORY VIEW 
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 min-h-screen">
      <div className="mb-6">
        <button 
            onClick={onBack}
            className="flex items-center gap-2 text-gray-500 hover:text-gray-900 transition-colors mb-4 font-medium"
        >
            <ArrowLeft className="w-5 h-5" /> Back to Home
        </button>

        <h1 className="text-3xl font-bold text-gray-900 mb-2">Discover Products</h1>
        <p className="text-gray-500">Explore our entire catalog with AI-driven insights.</p>
        
        <div className="flex overflow-x-auto gap-2 mt-6 pb-2 scrollbar-hide border-b border-gray-100">
            <button onClick={() => setActiveFilter('All')} className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors border ${activeFilter === 'All' ? 'bg-gray-900 text-white border-gray-900' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'}`}>All Categories</button>
            {Object.keys(groupedData).map(cat => (
                <button key={cat} onClick={() => setActiveFilter(cat)} className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors border ${activeFilter === cat ? 'bg-gray-900 text-white border-gray-900' : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'}`}>{cat}</button>
            ))}
        </div>
      </div>

      <FilterToolbar />

      <div className="space-y-12">
        {filteredCategories.length === 0 ? (
            <div className="text-center py-20 bg-gray-50 rounded-3xl border border-gray-100 border-dashed">
                <div className="text-4xl mb-4">🌪️</div>
                <h3 className="text-xl font-bold text-gray-900">No products found</h3>
                <p className="text-gray-500">Try adjusting your filters to see more results.</p>
                <button onClick={() => setFilters({ minRating: false, highSentiment: false, priceRange: 'all' })} className="mt-6 px-6 py-2 bg-white border border-gray-200 text-gray-700 font-bold rounded-full shadow-sm hover:shadow-md transition-all">Clear Filters</button>
            </div>
        ) : (
            filteredCategories.map(cat => (
                <section key={cat} className="space-y-6 animate-fade-in">
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 flex flex-col lg:flex-row items-stretch justify-between gap-6">
                        <div className="flex-1 flex flex-col justify-center">
                            <h2 className="text-3xl font-bold text-gray-900">{cat}</h2>
                            <p className="text-gray-500 mt-2 text-lg">{groupedData[cat].count} Item(s) {filters.highSentiment || filters.minRating || filters.priceRange !== 'all' ? ' (Filtered)' : ' Loaded'}</p>
                            <div className="flex items-center gap-4 mt-6">
                                <div className="bg-yellow-50 px-4 py-2 rounded-xl border border-yellow-100 flex items-center gap-3">
                                    <Star className="w-5 h-5 fill-yellow-500 text-yellow-500" />
                                    <div><span className="font-bold text-yellow-800 text-lg">{groupedData[cat].avgRating.toFixed(1)}</span><span className="text-[12px] text-yellow-600 uppercase font-medium tracking-wide"> Avg Rating</span></div>
                                </div>
                            </div>
                        </div>
                        <div className="flex flex-col sm:flex-row items-stretch gap-6 w-full lg:w-auto h-auto sm:h-60 lg:h-52">
                            <div className="flex-1 sm:w-64"><RatingDistributionChart distribution={groupedData[cat].ratingDist} /></div>
                            <div className="flex-1 sm:w-64"><EnhancedGauge score={groupedData[cat].avgSentiment} confidence={0.92} showConfidence={false} showPie={false} /></div>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
                        {groupedData[cat].items.slice(0, getVisibleCount(cat)).map(product => (
                            <ProductCard key={product.id} product={product} onClick={() => onProductClick(product)} />
                        ))}
                    </div>
                    {groupedData[cat].items.length > getVisibleCount(cat) && (
                        <div className="text-center pt-4 border-t border-gray-100">
                            <button onClick={() => handleShowMore(cat)} className="group inline-flex items-center gap-2 px-6 py-2 bg-gray-50 text-gray-600 font-bold rounded-full hover:bg-gray-100 transition-all text-sm">
                                {activeFilter === 'All' ? `View All ${cat} (${groupedData[cat].count - getVisibleCount(cat)} more)` : 'Load More Products'}
                                <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                            </button>
                        </div>
                    )}
                </section>
            ))
        )}
      </div>
    </div>
  );
};