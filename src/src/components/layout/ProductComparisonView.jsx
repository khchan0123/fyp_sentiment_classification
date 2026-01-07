import React, { useEffect, useState } from 'react';
import { useStore } from '../../context/StoreContext';
import { X, Star, ShieldCheck, AlertTriangle, ThumbsUp, ThumbsDown, Scale, ArrowLeft } from 'lucide-react'; 
import ProductImage from '../product/ProductImage';
import { toast } from 'sonner';

export const ProductComparisonView = ({ onBack }) => {
  const { compareList, removeFromCompare, clearCompare, addToCart, setSelectedProduct } = useStore();
  const [aiInsights, setAiInsights] = useState({});

  useEffect(() => {
    const fetchInsights = async () => {
      const insights = {};
      for (const product of compareList) {
        if (!product.keywords?.positive?.length) {
            try {
                const res = await fetch('http://localhost:5000/api/analyze-sentiment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ product_id: product.id, product_name: product.name })
                });
                const data = await res.json();
                insights[product.id] = data;
            } catch (e) {
                console.error(e);
            }
        }
      }
      setAiInsights(insights);
    };

    if (compareList.length > 0) {
        fetchInsights();
    }
  }, [compareList]);

  if (compareList.length === 0) {
    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
            <button 
                onClick={onBack}
                className="flex items-center gap-2 text-gray-500 hover:text-gray-900 transition-colors mb-8 font-medium"
            >
                <ArrowLeft className="w-5 h-5" /> Back to Home
            </button>

            <div className="text-center py-12">
                <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Scale className="w-10 h-10 text-gray-400" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900">Compare Products</h2>
                <p className="text-gray-500 mt-2">Add items from the product details page to compare them side-by-side.</p>
            </div>
        </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
        
        <button 
            onClick={onBack}
            className="flex items-center gap-2 text-gray-500 hover:text-gray-900 transition-colors mb-6 font-medium"
        >
            <ArrowLeft className="w-5 h-5" /> Back to Home
        </button>

        <div className="flex items-center justify-between mb-8">
            <div>
                <h2 className="text-3xl font-bold text-gray-900">Product Comparison</h2>
                <p className="text-gray-500">Side-by-side analysis using AI Truth Scores.</p>
            </div>
            <button 
                onClick={clearCompare}
                className="text-sm text-red-500 font-medium hover:bg-red-50 px-4 py-2 rounded-lg transition-colors"
            >
                Clear All
            </button>
        </div>

        {/* COMPARISON GRID */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-stretch">
            {compareList.map(product => {
                const sentiment = product.sentimentScore || 0.5;
                const score = Math.round(sentiment * 100);
                const insights = aiInsights[product.id] || product.keywords || { positive: [], negative: [] };
                
                const ratingValue = parseFloat(product.rating || product.starRating || 0);
                const ratingCountValue = product.ratingCount ? product.ratingCount.toLocaleString() : "0";

                const isMisleading = ratingValue > 4.0 && sentiment < 0.4;

                const badgeGradient = sentiment >= 0.8 
                    ? 'from-[#A7D397] to-[#8BC986]' 
                    : sentiment >= 0.4 
                        ? 'from-[#FCD34D] to-[#F59E0B]' 
                        : 'from-[#FF6B6B] to-[#FF4757]';

                return (
                    <div 
                        key={product.id} 
                        onClick={() => setSelectedProduct(product)}
                        className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden flex flex-col relative h-full cursor-pointer group hover:border-[#3ABEF9] transition-all hover:shadow-lg"
                    >
                        <button 
                            onClick={(e) => {
                                e.stopPropagation();
                                removeFromCompare(product.id);
                            }}
                            className="absolute top-3 right-3 p-1.5 bg-white/80 backdrop-blur rounded-full hover:bg-red-100 hover:text-red-500 transition-colors z-20 shadow-sm"
                        >
                            <X className="w-4 h-4" />
                        </button>

                        <div className="h-64 w-full bg-gray-50 p-6 relative border-b border-gray-100 flex items-center justify-center">
                            <div className="w-full h-full">
                                <ProductImage 
                                    src={product.image} 
                                    alt={product.name} 
                                    category={product.category} 
                                    className="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform duration-300" 
                                />
                            </div>
                            
                            <div className={`absolute bottom-4 left-4 px-3 py-1.5 rounded-full text-white text-xs font-bold flex items-center gap-1 shadow-lg bg-gradient-to-r ${badgeGradient}`}>
                                <ShieldCheck className="w-3 h-3" />
                                {score}% Quality
                            </div>

                            {isMisleading && (
                                <div className="absolute top-4 left-4 bg-gradient-to-r from-[#FF8E4E] to-[#FF6B6B] text-white px-3 py-1 rounded-full text-[10px] font-bold shadow-lg flex items-center gap-1 z-10 animate-pulse">
                                    <AlertTriangle className="w-3 h-3" />
                                    <span>Misleading Rating</span>
                                </div>
                            )}
                        </div>

                        {/* Content */}
                        <div className="p-6 flex flex-col flex-grow">
                            <h3 className="font-bold text-gray-900 mb-2 line-clamp-2 h-12 group-hover:text-[#3ABEF9] transition-colors">{product.name}</h3>
                            
                            <div className="flex items-center justify-between mb-6">
                                <span className="text-2xl font-bold text-gray-900">₹{product.price?.toLocaleString('en-IN')}</span>
                                <div className="flex items-center gap-1 flex-shrink-0">
                                    <Star className="w-4 h-4 text-[#FFD700] fill-[#FFD700]" />
                                    <span className="text-sm font-bold text-gray-900">{ratingValue}</span>
                                    <span className="text-xs text-gray-500 font-normal">({ratingCountValue})</span>
                                </div>
                            </div>

                            <div className="space-y-4 mb-6 flex-grow">
                                {/* Positive signals */}
                                <div>
                                    <h4 className="text-xs font-bold text-emerald-600 uppercase mb-2 flex items-center gap-1">
                                        <ThumbsUp className="w-3 h-3" /> Positive Signals
                                    </h4>
                                    <div className="flex flex-wrap gap-2">
                                        {(insights.positive || []).slice(0, 3).map((k, i) => (
                                            <span key={i} className="text-xs bg-emerald-50 text-emerald-700 px-2 py-1 rounded-md border border-emerald-100">
                                                {k}
                                            </span>
                                        ))}
                                        {(!insights.positive?.length) && <span className="text-xs text-gray-400">Analyzing...</span>}
                                    </div>
                                </div>

                                {/* Negative signals */}
                                <div>
                                    <h4 className="text-xs font-bold text-rose-600 uppercase mb-2 flex items-center gap-1">
                                        <ThumbsDown className="w-3 h-3" /> Potential Drawbacks
                                    </h4>
                                    <div className="flex flex-wrap gap-2">
                                        {(insights.negative || []).slice(0, 3).map((k, i) => (
                                            <span key={i} className="text-xs bg-rose-50 text-rose-700 px-2 py-1 rounded-md border border-rose-100">
                                                {k}
                                            </span>
                                        ))}
                                        {(!insights.negative?.length) && <span className="text-xs text-gray-400">Analyzing...</span>}
                                    </div>
                                </div>
                            </div>

                            <button 
                                onClick={(e) => {
                                    e.stopPropagation();
                                    const success = addToCart(product);
                                    if (success) toast.success(`Added ${product.name} to cart`, { icon: '🎉' });
                                }}
                                className="w-full py-3 rounded-xl bg-gradient-to-r from-[#3ABEF9] to-[#2AA8E0] text-white font-bold hover:shadow-lg transition-all mt-auto"
                            >
                                Add to Cart
                            </button>
                        </div>
                    </div>
                );
            })}
        </div>
    </div>
  );
};