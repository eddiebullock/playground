import React from 'react';

const TrendVisualizer = ({ trend }) => {
  if (!trend) {
    return <p>No trend data available.</p>;
  }

  const isPositive = trend.includes("Positive");
  const isNegative = trend.includes("Negative");

  return (
    <div className="flex items-center justify-center">
      {isPositive && (
        <div className="flex items-center text-green-600">
          <span className="mr-2">📈</span>
          <p>{trend}</p>
        </div>
      )}
      {isNegative && (
        <div className="flex items-center text-red-600">
          <span className="mr-2">📉</span>
          <p>{trend}</p>
        </div>
      )}
      {!isPositive && !isNegative && (
        <div className="flex items-center text-gray-600">
          <span className="mr-2">📊</span>
          <p>{trend}</p>
        </div>
      )}
    </div>
  );
};

export default TrendVisualizer;
