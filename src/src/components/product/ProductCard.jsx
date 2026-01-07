import React from 'react';
import { Star, ShoppingCart, AlertTriangle } from 'lucide-react';
import { useStore } from '../../context/StoreContext';
import { toast } from 'sonner';
import ProductImage from './ProductImage';

export const ProductCard = ({ product, onClick }) => {
  const { addToCart } = useStore();

  const { 
    name, 
    category, 
    image, 
    price, 
    starRating, 
    ratingCount, 
    sentimentScore 
  } = product;

  const handleAddToCart = (e) => {
    e.stopPropagation();
    const success = addToCart(product);
    if (success) {
      toast.success(`Added ${name} to cart`);
    }
  };

  const isMisleading = starRating > 4.0 && sentimentScore < 0.4;

  const getSentimentColor = () => {
    if (sentimentScore >= 0.8) return 'from-[#A7D397] to-[#8BC986]'; 
    if (sentimentScore >= 0.4) return 'from-[#FCD34D] to-[#F59E0B]'; 
    return 'from-[#FF6B6B] to-[#FF4757]'; 
  };

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-2xl overflow-hidden hover:shadow-2xl transition-all duration-300 cursor-pointer group border border-gray-100 hover:border-[#3ABEF9] hover:-translate-y-1 h-full flex flex-col"
    >
      <div className="relative aspect-square bg-gray-50 rounded-xl overflow-hidden mb-3 p-4 flex items-center justify-center">
        <ProductImage 
          src={image}     
          alt={name}      
          category={category} 
          className="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform duration-300" 
        />
        
        {/* Sentiment Badge */}
        <div className="absolute top-3 right-3">
          <div className={`bg-gradient-to-r ${getSentimentColor()} text-white px-3 py-1.5 rounded-full shadow-lg backdrop-blur-sm`}>
            <span className="text-sm font-bold">{Math.round(sentimentScore * 100)}%</span>
          </div>
        </div>

        {/* Misleading Rating Alert */}
        {isMisleading && (
          <div className="absolute top-3 left-3 bg-gradient-to-r from-[#FF8E4E] to-[#FF6B6B] text-white px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1.5 animate-pulse">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span className="text-xs font-bold">Rating Discrepancy</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-5 space-y-3 flex-grow flex flex-col">
        {/* Category */}
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide line-clamp-1">
            {category}
          </span>
          <div className="flex items-center gap-1 flex-shrink-0">
            <Star className="w-4 h-4 text-[#FFD700] fill-[#FFD700]" />
            <span className="text-sm font-bold text-gray-900">{starRating}</span>
          </div>
        </div>

        {/* Product Name */}
        <h3 className="text-gray-900 font-semibold line-clamp-2 min-h-[3rem] leading-snug">
          {name}
        </h3>

        {/* Price and Sentiment Score */}
        <div className="flex items-center justify-between mt-auto pt-2">
          <div className="text-2xl font-bold text-gray-900">
            ₹{price?.toLocaleString('en-IN') || "0"}
          </div>
        </div>

        <p className="text-xs text-gray-500">
          {ratingCount ? ratingCount.toLocaleString() : "0"} reviews analyzed
        </p>

        {/* Add to Cart Button */}
        <button
          onClick={handleAddToCart}
          className="w-full py-3 bg-gradient-to-r from-[#3ABEF9] to-[#2AA8E0] text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-[#3ABEF9]/30 transition-all duration-200 flex items-center justify-center gap-2 group-hover:scale-[1.02]"
        >
          <ShoppingCart className="w-4 h-4" />
          <span>Add to Cart</span>
        </button>
      </div>
    </div>
  );
};