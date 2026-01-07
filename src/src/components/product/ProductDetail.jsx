import React, { useEffect, useState, useRef } from 'react';
import { X, Star, ShoppingCart, AlertTriangle, Loader2, Info, Scale } from 'lucide-react';
import { useStore } from '../../context/StoreContext';
import { toast } from 'sonner';
import ProductImage from './ProductImage';
import { EnhancedGauge, AISummaryBox } from '../analytics/SentimentChart';

export const ProductDetail = ({ product: propProduct, onClose }) => {
  const { 
    products, 
    selectedProduct, 
    setSelectedProduct,
    addToCart, 
    addToCompare,
    compareList,
    fetchRecommendations, 
    recommendations, 
    isRecLoading, 
    currentUser 
  } = useStore();
  
  const activeProduct = selectedProduct || propProduct;

  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [aiData, setAiData] = useState(null);
  const scrollContainerRef = useRef(null);

  const handleClose = () => {
    setSelectedProduct(null); 
    onClose();               
  };

  const handleRecClick = (rec) => {
    const fullProduct = products.find(p => p.id === rec.id);
    setSelectedProduct(fullProduct || rec);
  };

  useEffect(() => {
    if (activeProduct?.id) {
        if (scrollContainerRef.current) scrollContainerRef.current.scrollTop = 0;
        setAiData(null); 
        fetchRecommendations(activeProduct.id);
        fetchAISummary(activeProduct.id, activeProduct.name);
    }
  }, [activeProduct]);

  const fetchAISummary = async (id, name) => {
    setIsLoadingAI(true);
    try {
        const response = await fetch('http://localhost:5000/api/analyze-sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_id: id, product_name: name })
        });
        
        if (response.ok) {
            const data = await response.json();
            setAiData(data);
        } else {
            setAiData({ positive: [], negative: [], meta: {} });
        }
    } catch (error) {
        setAiData(null);
    } finally {
        setIsLoadingAI(false);
    }
  };

  const handleAddToCart = () => {
    const success = addToCart(activeProduct);
    if (success) toast.success(`Added ${activeProduct.name} to cart`, { icon: '🎉' });
  };

  if (!activeProduct) return null;

  const ratingCount = activeProduct.ratingCount || 0;
  const sentimentScore = activeProduct.sentimentScore || 0.5;
  const modelConfidence = activeProduct.modelConfidence || 0.85; 
  const displayRating = activeProduct.starRating || activeProduct.rating || 0; 
  const displayDescription = activeProduct.description || activeProduct.about_product;
  const isHighRisk = sentimentScore < 0.4;
  const isMisleading = activeProduct.starRating > 4.0 && sentimentScore < 0.4;
  const isInCompare = compareList.some(p => p.id === activeProduct.id);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm font-sans animate-fade-in">
      <div className="bg-white rounded-3xl shadow-2xl max-w-7xl w-full max-h-[95vh] overflow-hidden animate-slide-in-up">
        
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-[#3ABEF9] to-[#FF8E4E] p-6 flex items-center justify-between z-10">
          <div className="flex items-center gap-4">
            <div className={`px-4 py-2 rounded-full ${
              sentimentScore >= 0.8 ? 'bg-gradient-to-r from-[#A7D397] to-[#8BC986]' : 
              sentimentScore >= 0.4 ? 'bg-gradient-to-r from-[#FCD34D] to-[#F59E0B]' : 
              'bg-gradient-to-r from-[#FF6B6B] to-[#FF4757]'
            } text-white font-bold shadow-lg`}>
              {Math.round(sentimentScore * 100)}% Truth Score
            </div>
            {isMisleading && (
              <div className="px-4 py-2 rounded-full bg-white/20 backdrop-blur-sm text-white font-medium flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                <span>Rating Discrepancy</span>
              </div>
            )}
          </div>
          <button onClick={handleClose} className="text-white hover:bg-white/20 rounded-full p-2 transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div ref={scrollContainerRef} className="grid lg:grid-cols-2 gap-0 overflow-auto max-h-[calc(95vh-88px)]">
          
          <div className="p-8 space-y-6 bg-white">
            <div className="relative aspect-square rounded-3xl overflow-hidden bg-gray-50 flex items-center justify-center p-8 border border-gray-100">
              <ProductImage 
                  src={activeProduct.image} 
                  alt={activeProduct.name} 
                  category={activeProduct.category} 
                  className="w-full h-full object-contain mix-blend-multiply drop-shadow-xl" 
              />
            </div>
            <div className="space-y-4">
               <div>
                  <span className="text-sm font-medium text-gray-500 uppercase tracking-wide">{activeProduct.category}</span>
                  <h2 className="text-3xl font-bold text-gray-900 mt-1">{activeProduct.name}</h2>
               </div>
               
               <div className="flex items-center justify-between py-4 border-y border-gray-200">
                  <div className="text-4xl font-bold text-gray-900">₹{activeProduct.price?.toLocaleString('en-IN')}</div>
                  <div className="flex items-center gap-2 bg-gradient-to-r from-[#FFD700] to-[#FFA500] px-4 py-2 rounded-full">
                    <Star className="w-5 h-5 text-white fill-white" />
                    <span className="text-xl font-bold text-white">{displayRating}</span>
                    <span className="text-sm text-white/80">({ratingCount.toLocaleString()})</span>
                  </div>
               </div>

               <div className="p-4 bg-gray-50 rounded-xl">
                  <p className="text-gray-700 leading-relaxed">{displayDescription || "No description available."}</p>
               </div>
               
               <div className="flex gap-3">
                   <button 
                     onClick={() => addToCompare(activeProduct)}
                     disabled={isInCompare}
                     className={`flex-1 py-4 rounded-xl font-bold border-2 transition-all flex items-center justify-center gap-2 ${
                        isInCompare 
                        ? 'border-gray-200 text-gray-400 bg-gray-50 cursor-not-allowed' 
                        : 'border-[#3ABEF9] text-[#3ABEF9] hover:bg-blue-50'
                     }`}
                   >
                      <Scale className="w-5 h-5" />
                      <span>{isInCompare ? 'Added to Compare' : 'Compare'}</span>
                   </button>

                   <button 
                     onClick={() => handleAddToCart()} 
                     className={`flex-[2] py-4 rounded-xl font-bold text-white text-lg shadow-lg transition-all hover:shadow-xl flex items-center justify-center gap-3 ${isHighRisk ? 'bg-gradient-to-r from-rose-400 to-rose-600' : 'bg-gradient-to-r from-[#3ABEF9] to-[#2AA8E0]'}`}
                   >
                      <ShoppingCart className="w-5 h-5" />
                      <span>{isHighRisk ? 'Buy with Caution' : 'Add to Cart'}</span>
                   </button>
               </div>

            </div>
          </div>

          <div className="p-8 bg-gradient-to-br from-gray-50 to-blue-50 flex flex-col gap-8">
            <div className="text-center">
              <h3 className="text-2xl font-bold text-gray-900">AI Analytics Dashboard</h3>
              <p className="text-gray-600 mt-1">Based on {ratingCount.toLocaleString()} reviews</p>
            </div>

            <div className="grid grid-cols-2 gap-4 auto-rows-fr min-h-[300px]">
                <div className="h-full">
                    <EnhancedGauge score={sentimentScore} confidence={modelConfidence} />
                </div>
                <div className="h-full">
                    {isLoadingAI ? (
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-full flex flex-col items-center justify-center text-center">
                            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mb-3" />
                            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Gemini Thinking...</p>
                        </div>
                    ) : (
                        <AISummaryBox 
                            keywords={{
                                positive: aiData?.positive || [], 
                                negative: aiData?.negative || []
                            }} 
                        />
                    )}
                </div>
            </div>

            <div className="bg-gray-900 rounded-2xl p-6 shadow-xl text-white">
               <div className="flex items-center gap-2 mb-4">
                 <Info className="w-5 h-5 text-blue-400" />
                 <h4 className="text-lg font-bold text-white">Transparency Report</h4>
               </div>
               <div className="space-y-4 text-sm">
                  <p className="text-gray-300 leading-relaxed">
                    This sentiment analysis uses a hybrid AI approach. A custom machine learning model classifies and calculates the overall numerical sentiment score.
                    AI is being used to summarize review with sentimental signals.
                  </p>
                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div>
                        <span className="block text-xs text-gray-500 uppercase font-bold mb-1">Sentiment Scoring Model</span>
                        <span className="font-medium text-blue-300">{aiData?.meta?.scoring_model || 'Loading...'}</span>
                    </div>
                    <div>
                        <span className="block text-xs text-gray-500 uppercase font-bold mb-1">Summary Model</span>
                        <span className="font-medium text-green-300">{aiData?.meta?.summary_model || 'Loading...'}</span>
                    </div>
                    <div>
                        <span className="block text-xs text-gray-500 uppercase font-bold mb-1">Dataset</span>
                        <span className="font-medium">{aiData?.meta?.dataset || 'Amazon Sales Data'}</span>
                    </div>
                    <div>
                        <span className="block text-xs text-gray-500 uppercase font-bold mb-1">Reviews Analyzed</span>
                        <span className="font-medium">{ratingCount.toLocaleString()}</span>
                    </div>
                  </div>
               </div>
            </div>

            <div className="border-t border-gray-200 pt-8">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                        ✨ {currentUser?.name ? currentUser.name.split(' ')[0] : 'You'} May Like
                    </h4>
                </div>

                {isRecLoading ? (
                    <div className="flex flex-col items-center justify-center py-8 text-gray-400 animate-pulse">
                        <Loader2 className="w-8 h-8 animate-spin mb-2 text-[#3ABEF9]" />
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-3">
                        {recommendations.slice(0, 4).map((rec) => (
                            <div 
                                key={rec.id} 
                                onClick={() => handleRecClick(rec)}
                                className="group relative bg-white border border-gray-100 rounded-xl p-3 hover:shadow-lg hover:border-[#3ABEF9] transition-all cursor-pointer"
                            >
                                <div className="aspect-square bg-gray-50 rounded-lg mb-2 overflow-hidden">
                                    <ProductImage src={rec.image} alt={rec.name} category={rec.category} className="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform" />
                                </div>
                                <h5 className="font-semibold text-gray-900 text-sm line-clamp-2 mb-1 leading-tight">{rec.name}</h5>
                                <div className="flex items-center justify-between">
                                    <span className="text-[#3ABEF9] font-bold text-sm">₹{rec.price?.toLocaleString('en-IN')}</span>
                                    <div className="flex items-center gap-0.5">
                                        <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                                        <span className="text-xs font-medium text-gray-600">{rec.starRating}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};