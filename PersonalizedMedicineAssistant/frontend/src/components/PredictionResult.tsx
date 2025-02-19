import React from "react";

const PredictionResult: React.FC<{ disease: string }> = ({ disease }) => {
  return (
    <div className="mt-4 p-3 bg-green-200 text-green-800 rounded-md">
      <h3>Predicted Disease:</h3>
      <p className="font-bold">{disease}</p>
    </div>
  );
};

export default PredictionResult;
