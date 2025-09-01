import React from "react";

const HomePage = ({ onSelect }) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-8 bg-gradient-to-br from-blue-500 via-purple-600 to-purple-800 p-6">
      <div className="bg-white/95 backdrop-blur-lg rounded-2xl shadow-2xl p-8 text-center max-w-lg w-full">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">
          Welcome to TCC Writing Analysis
        </h1>
        <p className="text-gray-600 mb-6">
          Select the type of analysis you would like to perform:
        </p>

        <select
          onChange={(e) => onSelect(e.target.value)}
          className="w-full p-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          defaultValue=""
        >
          <option value="" disabled>
            What type of analysis would you like to do?
          </option>
          <option value="keyness">Keyness</option>
          <option value="sentiment">Sentiment</option>
          <option value="clustering">Clustering</option>
          <option value="sensorimotor">Sensorimotor Norms</option>
        </select>
      </div>
    </div>
  );
};

export default HomePage;

