import React from 'react';

export const Button = ({ 
  children, 
  onClick, 
  variant = 'primary', 
  disabled = false, 
  className = '',
  type = 'button'
}) => {
  const baseClasses = 'px-4 py-2 rounded transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-slate-900 text-white hover:bg-slate-700 border border-slate-900',
    danger: 'bg-rose-600 text-white hover:bg-rose-700 border border-rose-600',
    success: 'bg-emerald-500 text-white hover:bg-emerald-600 border border-emerald-500',
    outline: 'border border-slate-900 text-slate-900 hover:bg-slate-900 hover:text-white',
    ghost: 'text-slate-900 hover:bg-slate-100',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
};
