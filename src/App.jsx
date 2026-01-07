import React, { useState } from 'react';
import { Toaster } from 'sonner';
import { StoreProvider, useStore } from './src/context/StoreContext';
import { Navbar } from './src/components/layout/Navbar';
import { Cart } from './src/components/layout/Cart';
import { ProductCard } from './src/components/product/ProductCard';
import { ProductDetail } from './src/components/product/ProductDetail';
import { SafetyWarningModal } from './src/components/common/Modal';
import { HomeView } from './src/components/layout/HomeView';
import { ProductDiscoverView } from './src/components/layout/ProductDiscoveryView';
import { ProductComparisonView } from './src/components/layout/ProductComparisonView';

function AppContent() {
  const { 
    products,
    showSafetyModal, 
    pendingPurchase, 
    confirmPurchase, 
    cancelPurchase, 
    selectedProduct, 
    setSelectedProduct,
    performSearch
  } = useStore();

  const [showCart, setShowCart] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const [currentView, setCurrentView] = useState('home');

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      <Navbar 
        onViewCart={() => setShowCart(true)} 
        searchQuery={searchQuery}
        setSearchQuery={(query) => {
            setSearchQuery(query);
            
            if (!query.trim()) {
                performSearch('');
            }
            
            if (query && currentView !== 'discover') {
                setCurrentView('discover');
            }
        }}

        onSearchFocus={() => {
            if (currentView !== 'discover') { 
                setCurrentView('discover');
            }
        }}
        
        onHomeClick={() => {
            setSearchQuery('');
            performSearch(''); 
            setCurrentView('home');
            window.scrollTo(0,0);
        }}
        
        onCompareClick={() => {
            setCurrentView('compare');
            window.scrollTo(0,0);
        }}
      />

      <div className="pt-20 pb-12">
        
        <>
            {currentView === 'home' && (
            <HomeView 
                onProductClick={setSelectedProduct} 
                onViewAllClick={() => {
                    setCurrentView('discover');
                    window.scrollTo(0, 0); 
                }}
            />
            )}

            {currentView === 'discover' && (
            <ProductDiscoverView 
                onProductClick={setSelectedProduct} 
                
                onBack={() => {
                    setSearchQuery(''); 
                    performSearch('');
                    setCurrentView('home');
                    window.scrollTo(0, 0);
                }}

                onClearSearch={() => {
                    setSearchQuery('');
                    performSearch(''); 
                }}
            />
            )}

            {currentView === 'compare' && (
            <ProductComparisonView 
                onBack={() => {
                    setCurrentView('home');
                    window.scrollTo(0, 0);
                }} 
            />
            )}
        </>

      </div>

      {selectedProduct && (
        <ProductDetail
          product={selectedProduct} 
          onClose={() => setSelectedProduct(null)}
        />
      )}

      <Cart isOpen={showCart} onClose={() => setShowCart(false)} />

      <SafetyWarningModal
        isOpen={showSafetyModal}
        onClose={cancelPurchase}
        onConfirm={confirmPurchase}
        product={pendingPurchase}
      />

      <Toaster position="bottom-center" richColors />
    </div>
  );
}

export default function App() {
  return (
    <StoreProvider>
      <AppContent />
    </StoreProvider>
  );
}