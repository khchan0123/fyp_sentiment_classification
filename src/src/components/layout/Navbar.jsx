import React, { useState } from 'react';
import { Search, ShoppingCart, ChevronDown, Sparkles, Loader2, Scale } from 'lucide-react';
import { useStore } from '../../context/StoreContext';
import { toast } from 'sonner';

export const Navbar = ({ onViewCart, searchQuery, setSearchQuery, onSearchFocus, onHomeClick, onCompareClick }) => {
  const { currentUser, switchUser, userProfiles, cartItemCount, loading, compareList, performSearch } = useStore();
  
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [debounceTimer, setDebounceTimer] = useState(null);

  // Handle Search
  const handleSearchChange = (e) => {
    const query = e.target.value;
    
    setSearchQuery(query);

    if (debounceTimer) clearTimeout(debounceTimer);

    const newTimer = setTimeout(() => {
        performSearch(query);
    }, 500);

    setDebounceTimer(newTimer);
  };

  const handleUserSwitch = (userId) => {
    const user = userProfiles.find((u) => u.id === userId);
    switchUser(userId);
    setShowUserMenu(false);
    
    if (userId !== 'guest') {
      toast.success(`Profile Loaded: ${user.description}`, {
        duration: 3000,
      });
    }
  };

  return (
    <nav className="bg-white sticky top-0 z-50 shadow-sm border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          
          {/* Logo */}
          <button 
            onClick={onHomeClick} 
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="w-12 h-12 bg-gradient-to-br from-[#3ABEF9] to-[#FF8E4E] rounded-2xl flex items-center justify-center shadow-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-[#3ABEF9] to-[#FF8E4E] bg-clip-text text-transparent">
              Shopeeps
            </h1>
          </button>

          {/* Search Bar */}
          <div className="flex-1 max-w-2xl mx-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search for more products..."
                value={searchQuery}
                onChange={handleSearchChange} 
                onClick={onSearchFocus}
                className="w-full pl-12 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-[#3ABEF9] focus:border-transparent transition-all text-gray-700 placeholder-gray-400"
              />
            </div>
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-4">
            <button
              onClick={onCompareClick}
              className="relative p-3 rounded-full hover:bg-gray-50 transition-colors group"
              title="Compare Products"
            >
              <Scale className="w-6 h-6 text-gray-700 group-hover:text-[#3ABEF9] transition-colors" />
              {compareList && compareList.length > 0 && (
                <span className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-r from-[#FF8E4E] to-[#FF6B6B] text-white text-xs rounded-full flex items-center justify-center font-bold shadow-lg">
                  {compareList.length}
                </span>
              )}
            </button>
            
            <button
              onClick={onViewCart}
              className="relative p-3 rounded-full hover:bg-gray-50 transition-colors group"
            >
              <ShoppingCart className="w-6 h-6 text-gray-700 group-hover:text-[#3ABEF9] transition-colors" />
              {cartItemCount > 0 && (
                <span className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-r from-[#FF8E4E] to-[#FF6B6B] text-white text-xs rounded-full flex items-center justify-center font-bold shadow-lg">
                  {cartItemCount}
                </span>
              )}
            </button>

            {loading || !currentUser ? (
               <div className="w-40 h-10 bg-gray-100 rounded-full animate-pulse flex items-center justify-center text-gray-400 text-sm">
                 <Loader2 className="w-4 h-4 animate-spin mr-2" />
                 Loading...
               </div>
            ) : (
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-3 px-4 py-2 rounded-full bg-gray-50 hover:bg-gray-100 transition-all"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#3ABEF9] to-[#A7D397] flex items-center justify-center text-white font-bold text-sm">
                    {currentUser?.name?.charAt(0) || '?'}
                  </div>
                  <span className="text-gray-700 font-medium">
                    {currentUser?.name || 'Guest'}
                  </span>
                  <ChevronDown className="w-4 h-4 text-gray-500" />
                </button>

                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-80 bg-white rounded-2xl shadow-2xl overflow-hidden z-50 border border-gray-100">
                    <div className="p-4 bg-gradient-to-r from-[#3ABEF9] to-[#FF8E4E] text-white">
                      <p className="text-sm font-medium">Switch Profile</p>
                    </div>
                    {userProfiles.map((profile) => (
                      <button
                        key={profile.id}
                        onClick={() => handleUserSwitch(profile.id)}
                        className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-b-0 ${
                          currentUser.id === profile.id ? 'bg-blue-50' : ''
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#3ABEF9] to-[#A7D397] flex items-center justify-center text-white font-bold">
                            {profile.name.charAt(0)}
                          </div>
                          <div>
                            <div className="text-gray-900 font-medium">{profile.name}</div>
                            <div className="text-gray-500 text-sm">{profile.description || 'No description'}</div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};