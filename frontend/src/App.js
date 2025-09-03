import React, { useState } from "react";
import HomePage from "./Components/HomePage";
import KeynessLanding from "./Components/Keyness/KeynessLanding";
import ClusteringLanding from "./Components/Clustering/ClusteringLanding";
import SentimentLanding from "./Components/Sentiment/SentimentLanding";
import SensorimotorLanding from "./Components/Sensorimotor/SensorimotorLanding";
import useUnloadCleanup from "./Hooks/useUnloadCleanup";

function App() {
  const [activePage, setActivePage] = useState("home");

  const handleBack = () => setActivePage("home");

  useUnloadCleanup();

  return (
    <div className="App p-6">
      {activePage === "home" && <HomePage onSelect={setActivePage} />}

      {activePage === "keyness" && <KeynessLanding onBack={handleBack} />}
      {activePage === "clustering" && <ClusteringLanding onBack={handleBack} />}
      {activePage === "sentiment" && <SentimentLanding onBack={handleBack} />}
      {activePage === "sensorimotor" && (
        <SensorimotorLanding onBack={handleBack} />
      )}
    </div>
  );
}

export default App;

