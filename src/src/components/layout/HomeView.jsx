import React, { useState } from 'react';
import { useStore } from '../../context/StoreContext';
import { ProductCard } from '../product/ProductCard'; 
import { Star, ShieldCheck, Sparkles, ChevronRight, Zap } from 'lucide-react';
import ProductImage from '../product/ProductImage';

export const HomeView = ({ onProductClick, onViewAllClick }) => {
  const { 
    currentUser, 
    homeRecommendations, 
    smartDiscoveryFeed, 
    isFeedLoading,
  } = useStore();
  
  const [activeCategory, setActiveCategory] = useState('All');

  const filteredProducts = activeCategory === 'All' 
    ? smartDiscoveryFeed 
    : smartDiscoveryFeed.filter(p => p.category === activeCategory);

  const categories = ['All', ...new Set(smartDiscoveryFeed.map(p => p.category))].slice(0, 6);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      
      {/* SECTION 1: HERO (Persona Based) */}
      <section className="bg-white rounded-3xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] rounded-xl text-white shadow-lg">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">
              {currentUser?.id === 'guest' ? 'Top 4 Big Deals' : `Top Picks for ${currentUser?.name}`}
            </h2>
            <p className="text-sm text-gray-500">
              {currentUser?.id === 'guest' ? 'Guest Special Offers Awaiting' : `Recommended based on your ${currentUser?.bias} history.`}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {homeRecommendations.map((item) => (
            <div key={item.id} onClick={() => onProductClick(item)} className="group block bg-gray-50 rounded-xl p-3 border border-transparent hover:border-[#3ABEF9] hover:shadow-md transition-all cursor-pointer">
              <div className="relative aspect-square bg-white rounded-lg overflow-hidden mb-3 p-3">
                <ProductImage src={item.image} alt={item.name} category={item.category} className="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform" />
                
                {item.discount > 0 && (
                    <div className="absolute top-2 right-2 bg-[#FF6B6B] text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full shadow-sm">-{item.discount}%</div>
                )}
                
                <div className={`absolute bottom-2 left-2 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full flex items-center gap-1 shadow-sm ${
                    item.sentimentScore >= 0.8 
                        ? 'bg-gradient-to-r from-[#A7D397] to-[#8BC986]' // Standard Green
                        : 'bg-gradient-to-r from-[#FCD34D] to-[#F59E0B]' // Standard Yellow
                }`}>
                  <ShieldCheck className="w-3 h-3" /> {Math.round(item.sentimentScore * 100)}%
                </div>
              </div>
              
              <div className="space-y-1">
                <h3 className="font-bold text-gray-900 text-sm line-clamp-2 min-h-[2.5em] leading-tight">{item.name}</h3>
                <div className="flex items-center justify-between pt-1">
                   <span className="text-sm font-bold text-[#ee4d2d]">₹{item.price?.toLocaleString('en-IN')}</span>
                   <span className="text-xs text-gray-500 flex items-center font-bold gap-1">
                    <Star className="w-4 h-4 text-[#FFD700] fill-[#FFD700]" /> {item.rating || item.starRating}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 2: DISCOVERY (Context Based) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-50 rounded-lg text-blue-500">
                    <Zap className="w-5 h-5" />
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-gray-900">Recommended For You</h2>
                    <p className="text-sm text-gray-500">Real-time suggestions based on your recent activity.</p>
                </div>
            </div>
        </div>

        {/* Categories */}
        <div className="flex overflow-x-auto gap-2 pb-2 scrollbar-hide">
          {categories.map(cat => (
            <button key={cat} onClick={() => setActiveCategory(cat)} className={`whitespace-nowrap px-5 py-2 rounded-full text-sm font-medium transition-colors ${activeCategory === cat ? 'bg-gray-900 text-white' : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
              {cat}
            </button>
          ))}
        </div>

        {/* Grid */}
        {isFeedLoading ? (
            <div className="py-20 flex flex-col items-center justify-center text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-[#3ABEF9] mb-4"></div>
                <p className="text-gray-500 font-medium">Personalizing your feed...</p>
            </div>
        ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredProducts.slice(0, 16).map(product => (
                <ProductCard key={product.id} product={product} onClick={() => onProductClick(product)} />
            ))}
            </div>
        )}

        <div className="pt-8 text-center border-t border-gray-100">
            <button onClick={onViewAllClick} className="group inline-flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-gray-900 to-gray-800 text-white font-bold rounded-2xl shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300">
                View All Products <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
        </div>
      </section>

    </div>
  );
};