import React from "react";

const App: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 text-center p-6">
      {/* Logo with smaller size */}
      <img src="/logo.webp" alt="Logo" className="w-20 h-20 mb-4" />

      {/* Title */}
      <h1 className="text-2xl font-bold text-gray-900">
        Personalized Medicine Assistant
      </h1>

      {/* Subtitle */}
      <p className="text-lg text-gray-600 mt-2">
        Welcome to your AI-powered healthcare assistant.
      </p>

      {/* Button */}
      <button className="mt-6 px-6 py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition">
        Check Symptoms
      </button>
    </div>
  );
};

export default App;
