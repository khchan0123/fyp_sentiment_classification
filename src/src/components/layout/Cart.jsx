import React from 'react';
import { X, Trash2, ShoppingCart, Sparkles } from 'lucide-react';
import { useStore } from '../../context/StoreContext';
import ProductImage from '../product/ProductImage';

export const Cart = ({ isOpen, onClose }) => {
  const { cart, cartTotal, removeFromCart, updateQuantity } = useStore();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-end">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      
      <div className="relative bg-white w-full sm:w-[420px] h-full sm:h-[95vh] sm:rounded-l-3xl shadow-2xl flex flex-col animate-slide-in-up">
        {/* Header */}
        <div className="bg-gradient-to-r from-[#3ABEF9] to-[#FF8E4E] p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center">
                <ShoppingCart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Your Cart</h2>
                <p className="text-white/80 text-sm">{cart.length} items</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:bg-white/20 rounded-full p-2 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Cart Items */}
        <div className="flex-1 overflow-auto p-6 space-y-4">
          {cart.length === 0 ? (
            <div className="text-center py-20">
              <div className="w-24 h-24 bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <ShoppingCart className="w-12 h-12 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Your cart is empty</h3>
              <p className="text-gray-500">Start adding some awesome products!</p>
            </div>
          ) : (
            cart.map((item) => (
              <div
                key={item.id}
                className="bg-gradient-to-br from-gray-50 to-white p-4 rounded-2xl border border-gray-100 hover:shadow-lg transition-all"
              >
                <div className="flex gap-4">
                  <div className="w-24 h-24 bg-gray-50 rounded-xl border border-gray-100 flex-shrink-0 p-2 flex items-center justify-center">
                    <ProductImage
                      src={item.image}
                      alt={item.name}
                      category={item.category}
                      className="w-full h-full object-contain mix-blend-multiply"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-gray-900 font-semibold text-sm line-clamp-2 pr-2">
                        {item.name}
                      </h3>
                      <button
                        onClick={() => removeFromCart(item.id)}
                        className="text-gray-400 hover:text-red-500 transition-colors p-1"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    
                    <div className={`mt-1 inline-flex px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide ${
                        item.sentimentScore >= 0.8 ? 'bg-emerald-50 text-emerald-600' : 
                        item.sentimentScore >= 0.4 ? 'bg-yellow-50 text-yellow-600' : 
                        'bg-rose-50 text-rose-600'
                    }`}>
                        {Math.round(item.sentimentScore * 100)}% Sentiment
                    </div>
                    
                    <div className="flex items-center justify-between mt-3">
                      <span className="text-2xl font-bold bg-gray-900 bg-clip-text text-transparent">
                        ₹{(item.price * item.quantity).toLocaleString('en-IN')}
                      </span>
                      
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => updateQuantity(item.id, item.quantity - 1)}
                          className="w-8 h-8 rounded-lg bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-700 font-bold transition-colors"
                        >
                          -
                        </button>
                        <span className="text-gray-900 font-bold w-8 text-center">
                          {item.quantity}
                        </span>
                        <button
                          onClick={() => updateQuantity(item.id, item.quantity + 1)}
                          className="w-8 h-8 rounded-lg bg-gradient-to-r from-[#3ABEF9] to-[#2AA8E0] hover:shadow-lg text-white font-bold transition-all"
                        >
                          +
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        {cart.length > 0 && (
          <div className="border-t border-gray-200 p-6 space-y-4 bg-white">
            <div className="space-y-2">
              <div className="flex justify-between items-center text-gray-600">
                <span>Subtotal</span>
                <span>₹{cartTotal.toLocaleString('en-IN')}</span>
              </div>
              <div className="flex justify-between items-center text-gray-600">
                <span>Shipping</span>
                <span className="text-[#A7D397] font-semibold">FREE</span>
              </div>
              <div className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent my-3" />
              <div className="flex justify-between items-center">
                <span className="text-lg font-bold text-gray-900">Total</span>
                <span className="text-2xl font-bold bg-gradient-to-r from-[#3ABEF9] to-[#FF8E4E] bg-clip-text text-transparent">
                  ₹{cartTotal.toLocaleString('en-IN')}
                </span>
              </div>
            </div>
            
            <button className="w-full py-4 bg-gradient-to-r from-[#3ABEF9] to-[#2AA8E0] text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2">
              <Sparkles className="w-5 h-5" />
              <span>Checkout</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};