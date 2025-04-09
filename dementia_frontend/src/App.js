import React, { useState } from "react";
import "./App.css";

const App = () => {
  const [formData, setFormData] = useState({
    gender: "M",
    age: "",
    EDUC: "",
    SES: "",
    MMSE: "",
    CDR: "",
    eTIV: "",
    nWBV: "",
    ASF: ""
  });

  const [selectedModel, setSelectedModel] = useState("ANN");
  const [prediction, setPrediction] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    const numericFields = ["age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"];
    const parsedValue = numericFields.includes(name)
      ? value === "" ? "" : parseFloat(value)
      : value;

    setFormData((prev) => ({ ...prev, [name]: parsedValue }));
    setPrediction("");
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    setPrediction("");
  };

  const handleSubmit = async () => {
    const payload = { ...formData, model_name: selectedModel };

    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      const predictionKey = `${selectedModel}_Prediction`;
      setPrediction(data[predictionKey]);
    } catch {
      setPrediction("Error fetching prediction");
    }
  };

  return (
    <div className="background">
      <h2 className="page-title">
        <span className="page-title">Dementia</span>
        <span className="page-title-2"> Detection</span>
      </h2>

      <div className="container">
        <div className="medical-form">
          <div className="form-group full-width">
            <label>Model</label>
            <select value={selectedModel} onChange={handleModelChange}>
              <option value="ANN">ANN</option>
              <option value="KNN">KNN</option>
              <option value="Logistic_Regression">Logistic Regression</option>
              <option value="Naive_Bayes">Naive Bayes</option>
              <option value="AdaBoost">AdaBoost</option>
              <option value="Random_Forest">Random Forest</option>
              <option value="Decision_Tree">Decision Tree</option>
            </select>
          </div>

          <div className="form-group full-width">
            <label>Gender</label>
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>

          {["age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"].map((field) => (
            <div key={field} className="form-group">
              <label>{field}</label>
              <input
                type="number"
                name={field}
                value={formData[field]}
                onChange={handleChange}
                required
              />
            </div>
          ))}

          <button type="button" className="submit-btn" onClick={handleSubmit}>
            Detect
          </button>
        </div>

        {prediction && (
          <h3 className={`prediction-result ${prediction === "Non-Demented" ? "non-demented" : "demented"}`}>
            {selectedModel} Prediction: {prediction}
          </h3>
        )}
      </div>
    </div>
  );
};

export default App;
