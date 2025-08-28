import React from "react";

const SentimentAnalyser = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-6">
      <div className="bg-white rounded-2xl shadow-lg p-10 max-w-2xl w-full text-center">
        <h1 className="text-3xl font-bold text-blue-600 mb-4">
          Sentiment Analysis
        </h1>
        <p className="text-gray-600">
          Analyse the emotional tone and polarity of your text.
        </p>
      </div>
    </div>
  );
};

export default SentimentAnalyser;
