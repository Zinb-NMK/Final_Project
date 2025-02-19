import React, { useState } from "react";
import axios from "axios";
import PredictionResult from "./PredictionResult";

const SymptomForm: React.FC = () => {
  const [symptoms, setSymptoms] = useState<string>("");
  const [prediction, setPrediction] = useState<string>("");

  const handleSubmit = async () => {
    const symptomList = symptoms.split(",").map((s) => s.trim());
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", { symptoms: symptomList });
      setPrediction(response.data.predicted_disease);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md w-1/2">
      <input
        type="text"
        className="border p-2 w-full"
        placeholder="Enter symptoms, comma-separated"
        value={symptoms}
        onChange={(e) => setSymptoms(e.target.value)}
      />
      <button className="mt-3 p-2 bg-blue-500 text-white" onClick={handleSubmit}>Predict</button>
      {prediction && <PredictionResult disease={prediction} />}
    </div>
  );
};

export default SymptomForm;
