import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip 
} from 'recharts';
import { ThumbsUp, ThumbsDown, Sparkles } from 'lucide-react';

const CHART_CONTAINER_CLASS = "bg-white rounded-2xl p-6 shadow-sm border border-gray-100 flex flex-col items-center h-full justify-between font-sans transition-all hover:shadow-md";
const CHART_TITLE_CLASS = "text-sm font-bold text-gray-600 uppercase tracking-widest w-full text-center mb-4";

// --- 1. ENHANCED GAUGE ---
export const EnhancedGauge = ({ score, confidence, showConfidence = true, showPie = true }) => {
  const posValue = Math.round(score * 100);
  const negValue = 100 - posValue;

  // High & Low Sentiment Score Logic
  const isHigh = score >= 0.8;
  const isModerate = score >= 0.4 && score < 0.8;

  const data = [
    { 
        name: 'Positive', 
        value: posValue, 
        color: isHigh ? '#10B981' : isModerate ? '#F59E0B' : '#F43F5E' 
    }, 
    { name: 'Negative', value: negValue, color: '#F3F4F6' } 
  ];

  let summaryText = "";
  let summaryColor = "";
  let footerLabel = "";

  if (isHigh) { 
      summaryText = "Excellent"; 
      summaryColor = "text-emerald-600"; 
      footerLabel = "High Positive Sentiment Rate from Reviews";
  } else if (isModerate) { 
      summaryText = "Moderate"; 
      summaryColor = "text-yellow-600"; 
      footerLabel = "Mixed / Average Review Sentiment";
  } else { 
      summaryText = "Critical Issues"; 
      summaryColor = "text-rose-600"; 
      footerLabel = "Low / Critical Sentiment Rate from Reviews";
  }

  return (
    <div className={CHART_CONTAINER_CLASS}>
      <h4 className={CHART_TITLE_CLASS}>
        True Quality Score
      </h4>

      <div className={`relative w-full flex items-end justify-center pb-2 ${showPie ? 'h-[150px]' : 'h-[100px]'}`}>
        {showPie ? (
            /* MODE A: FULL CHART */
            <>
                <div className="w-full h-full absolute inset-0">
                    <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                        data={data}
                        cx="50%"
                        cy="100%"
                        startAngle={180}
                        endAngle={0}
                        innerRadius="120%"
                        outerRadius="145%"
                        paddingAngle={2}
                        dataKey="value"
                        stroke="none"
                        cornerRadius={2}
                        >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                        </Pie>
                        <RechartsTooltip 
                        formatter={(value, name) => [`${value}%`, `${name}`]}
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                        />
                    </PieChart>
                    </ResponsiveContainer>
                </div>
                
                <div className="absolute bottom-0 left-0 right-0 flex flex-col items-center justify-end pb-2">
                    <div className={`text-4xl font-extrabold ${summaryColor} leading-none`}>
                    {posValue}%
                    </div>
                    <div className={`text-sm font-bold ${summaryColor} flex items-center gap-1 mt-1`}>
                        {isHigh && <Sparkles className="w-3 h-3" />}
                        {summaryText}
                    </div>
                </div>
            </>
        ) : (
            /* MODE B: TEXT ONLY */
            <div className="flex flex-col items-center justify-center">
                 <div className={`text-5xl font-extrabold ${summaryColor} leading-none tracking-tight`}>
                    {posValue}%
                 </div>
                 <div className={`text-lg font-bold ${summaryColor} flex items-center gap-2 mt-2`}>
                     {isHigh && <Sparkles className="w-5 h-5" />}
                     {summaryText}
                 </div>
            </div>
        )}
      </div>

      {/* FOOTER LABEL */}
      <p className={`text-[11px] mt-3 text-center font-bold ${summaryColor} opacity-90`}>
         {footerLabel}
      </p>

      {showConfidence && (
        <div className="w-full bg-gray-50 rounded-xl p-3 border border-gray-100 mt-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-gray-500 uppercase">Model Confidence</span>
            <span className="text-xs font-bold text-gray-700">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div 
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-1000" 
              style={{ width: `${confidence * 100}%` }} 
            />
          </div>
        </div>
      )}
    </div>
  );
};

// 2. RATING DISTRIBUTION CHART
export const RatingDistributionChart = ({ distribution }) => {
    const values = Object.values(distribution);
    const maxVal = Math.max(...values) || 1; 

    return (
        <div className={CHART_CONTAINER_CLASS}>
            <h4 className={CHART_TITLE_CLASS}>
                Rating Distribution
            </h4>
            
            <div className="flex items-end justify-between w-full flex-grow gap-3 px-2 min-h-[120px]">
                {[1, 2, 3, 4, 5].map((star) => {
                    const count = distribution[star] || 0;
                    const heightPercent = count === 0 ? 2 : (count / maxVal) * 100; 
                    
                    return (
                        <div key={star} className="flex flex-col items-center gap-2 group w-full h-full justify-end">
                            <div className="w-full bg-gray-50 rounded-md relative h-full max-h-[100px] flex items-end overflow-hidden group-hover:bg-gray-100 transition-colors">
                                <div 
                                    className={`w-full rounded-md transition-all duration-1000 ease-out ${
                                        star >= 4 ? 'bg-yellow-400' : star === 3 ? 'bg-yellow-200' : 'bg-gray-300'
                                    }`}
                                    style={{ height: `${heightPercent}%` }}
                                />
                            </div>
                            <div className="text-xs font-bold text-gray-500 flex items-center">
                                {star}<span className="text-[10px] ml-[1px] text-yellow-500">★</span>
                            </div>
                        </div>
                    );
                })}
            </div>
            
             <p className="text-[10px] text-gray-400 mt-1 text-center font-medium">
                 Total Reviews Analyzed
             </p>
        </div>
    );
};

// 3. AI SUMMARY BOX
export const AISummaryBox = ({ keywords }) => {
    const posList = keywords?.positive?.length > 0 
      ? keywords.positive.slice(0, 3) 
      : ["Analyzing...", "Wait a moment"];
      
  const negList = keywords?.negative?.length > 0 
      ? keywords.negative.slice(0, 3) 
      : ["Analyzing...", "Wait a moment"];

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-full flex flex-col font-sans">
      <h4 className={CHART_TITLE_CLASS}>
        AI-Generated Highlights
      </h4>
      
      <div className="flex flex-col gap-4 flex-grow justify-center">
        <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-100">
            <div className="flex items-center gap-2 mb-2">
                <ThumbsUp className="w-4 h-4 text-emerald-600" />
                <span className="text-xs font-bold text-emerald-800 uppercase">Positive Peeps</span>
            </div>
            <div className="flex flex-wrap gap-2">
                {posList.map((k, i) => (
                    <span key={i} className="px-2 py-1 bg-white text-emerald-700 text-xs font-medium rounded-md shadow-sm">
                        {k}
                    </span>
                ))}
            </div>
        </div>

        <div className="bg-rose-50 rounded-xl p-4 border border-rose-100">
            <div className="flex items-center gap-2 mb-2">
                <ThumbsDown className="w-4 h-4 text-rose-600" />
                <span className="text-xs font-bold text-rose-800 uppercase">Negative Peeps</span>
            </div>
            <div className="flex flex-wrap gap-2">
                {negList.map((k, i) => (
                    <span key={i} className="px-2 py-1 bg-white text-rose-700 text-xs font-medium rounded-md shadow-sm">
                        {k}
                    </span>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
};