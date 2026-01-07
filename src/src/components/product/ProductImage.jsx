import React, { useState, useEffect } from 'react';

const CATEGORY_FALLBACKS = {
  'Home&Kitchen': 'https://images.unsplash.com/photo-1659720879171-5fb849451fe4?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0',
  
  'Computers&Accessories': 'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=600&q=80',

  'Electronics': 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=600&q=80',
  
  'OfficeProducts': 'https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=600&q=80',
  
  'Toys&Games': 'https://images.unsplash.com/photo-1566576912321-d58ddd7a6088?auto=format&fit=crop&w=600&q=80',

  'MusicalInstruments': 'https://images.unsplash.com/photo-1556379118-7034d926d258?auto=format&fit=crop&w=600&q=80',

  'Health&PersonalCare': 'https://images.unsplash.com/photo-1576426863863-1fa093c36003?auto=format&fit=crop&w=600&q=80',

  'HomeImprovement': 'https://images.unsplash.com/photo-1581244277914-a91b7ac2deda?auto=format&fit=crop&w=600&q=80',

  'Car&Motorbike': '  https://images.unsplash.com/photo-1533630217389-3a5e4dff5683?q=80&w=1170&auto=format&fit=crop&w=600&q=80',

  'Default': 'https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=600&q=80'
};

const ProductImage = ({ src, alt, category, className, onClick }) => {
  const [imgSrc, setImgSrc] = useState(src);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setImgSrc(src);
    setHasError(false);
  }, [src]);

  const handleError = () => {
    if (!hasError) {
      const fallbackUrl = CATEGORY_FALLBACKS[category] || CATEGORY_FALLBACKS['Default'];
      setImgSrc(fallbackUrl);
      setHasError(true);
    }
  };

  return (
    <img
      src={imgSrc}
      alt={alt}
      onError={handleError}
      onClick={onClick}
      className={`${className} ${hasError ? 'opacity-90 grayscale-[10%]' : ''}`} 
      loading="lazy"
    />
  );
};

export default ProductImage;