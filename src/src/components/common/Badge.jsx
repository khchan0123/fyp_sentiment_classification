import React from 'react';
import { Circle } from 'lucide-react';

export const SentimentBadge = ({ score, size = 'md', showLabel = false }) => {
  const getSentimentConfig = (score) => {
    if (score >= 0.6) {
      return {
        color: 'text-emerald-600',
        bgColor: 'bg-emerald-600',
        label: 'Trusted',
      };
    } else if (score >= 0.4) {
      return {
        color: 'text-gray-600',
        bgColor: 'bg-gray-600',
        label: 'Mixed',
      };
    } else {
      return {
        color: 'text-rose-600',
        bgColor: 'bg-rose-600',
        label: 'Warning',
      };
    }
  };

  const config = getSentimentConfig(score);
  
  const iconSizes = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  return (
    <div className="inline-flex items-center gap-1">
      <Circle className={`${iconSizes[size]} ${config.bgColor} fill-current`} />
      {showLabel && (
        <span className={`${config.color} text-sm`}>
          {config.label}
        </span>
      )}
    </div>
  );
};

export const Badge = ({ children, variant = 'default', className = '' }) => {
  const variants = {
    default: 'bg-slate-200 text-slate-900',
    positive: 'bg-emerald-100 text-emerald-700 border border-emerald-300',
    negative: 'bg-rose-100 text-rose-700 border border-rose-300',
    neutral: 'bg-gray-100 text-gray-700 border border-gray-300',
  };

  return (
    <span className={`inline-flex items-center px-2 py-1 rounded ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
};
