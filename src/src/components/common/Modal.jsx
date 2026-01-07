import React from 'react';
import { X, AlertTriangle, ShieldAlert } from 'lucide-react';

export const Modal = ({ isOpen, onClose, children, title }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white rounded-3xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-auto animate-slide-in-up">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-900">{title}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-900 transition-colors rounded-full p-2 hover:bg-gray-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );
};

export const SafetyWarningModal = ({ isOpen, onClose, onConfirm, product }) => {
  if (!product) return null;

  const negativeKeywords = product.keywords?.negative || [];
  
  const criticalFlags = negativeKeywords.filter((kw) => 
    kw.toLowerCase().includes('fire') ||
    kw.toLowerCase().includes('hazard') ||
    kw.toLowerCase().includes('explod') ||
    kw.toLowerCase().includes('danger') ||
    kw.toLowerCase().includes('scam') ||
    kw.toLowerCase().includes('fake')
  );

  const count = product.ratingCount || product.reviewCount || 0;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="">
      <div className="space-y-6">
        {/* Warning Header */}
        <div className="text-center">
          <div className="w-20 h-20 bg-gradient-to-r from-[#FF6B6B] to-[#FF4757] rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg">
            <ShieldAlert className="w-10 h-10 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Safety Warning</h2>
          <p className="text-gray-600">AI detected quality concerns with this product</p>
        </div>

        {/* Alert Box */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 border-2 border-[#FF6B6B] rounded-2xl p-5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-[#FF6B6B] flex-shrink-0 mt-0.5" />
            <p className="text-gray-700 text-sm leading-relaxed">
              Our AI analyzed <span className="font-bold text-gray-900">{count.toLocaleString()} verified reviews</span> and 
              found significant quality concerns. Please review the data before purchasing.
            </p>
          </div>
        </div>

        {/* Product Info */}
        <div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-sm">Product:</span>
            <span className="text-gray-900 font-semibold text-sm text-right max-w-[60%]">{product.name}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-sm">AI Truth Score:</span>
            <span className="text-[#FF6B6B] font-bold text-lg">
              {Math.round(product.sentimentScore * 100)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-sm">Model Confidence:</span>
            <span className="text-gray-900 font-semibold">
              {Math.round(product.modelConfidence * 100)}%
            </span>
          </div>
        </div>

        {/* Critical Flags */}
        {criticalFlags.length > 0 && (
          <div className="bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-2xl p-5">
            <p className="text-gray-900 font-bold mb-3 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-[#FF8E4E]" />
              Critical Mentions Detected:
            </p>
            <div className="flex flex-wrap gap-2">
              {criticalFlags.map((keyword, idx) => (
                <span
                  key={idx}
                  className="px-4 py-2 bg-gradient-to-r from-red-100 to-orange-100 text-[#991B1B] rounded-full text-sm font-semibold border border-red-200"
                >
                  "{keyword}"
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            className="flex-1 py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-bold hover:bg-gray-50 transition-all"
          >
            Cancel Purchase
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-3 bg-gradient-to-r from-[#FF6B6B] to-[#FF4757] text-white rounded-xl font-bold hover:shadow-lg transition-all"
          >
            Buy Anyway
          </button>
        </div>

        {/* Disclaimer */}
        <p className="text-gray-500 text-center text-xs leading-relaxed">
          This analysis is powered by AI sentiment analysis trained on millions of product reviews.
          Results are provided for informational purposes only.
        </p>
      </div>
    </Modal>
  );
};