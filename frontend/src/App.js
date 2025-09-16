import React, { useState } from "react";
import HomePage from "./Components/HomePage";
import KeynessLanding from "./Components/Keyness/KeynessLanding";
import ClusteringLanding from "./Components/Clustering/ClusteringLanding";
import SentimentLanding from "./Components/Sentiment/SentimentLanding";
import SensorimotorLanding from "./Components/Sensorimotor/SensorimotorLanding";
import useUnloadCleanup from "./Hooks/useUnloadCleanup";

function App() {
    const [activePage, setActivePage] = useState("home");
    const [selectedGenre, setSelectedGenre] = useState("");

    const handleBack = () => setActivePage("home");

    useUnloadCleanup();

    return (
        <div className="App p-6">
            {activePage === "home" && (
                <HomePage
                    onSelect={setActivePage}
                    selectedGenre={selectedGenre}        
                    onSelectGenre={setSelectedGenre}    
                />
            )}

            {activePage === "keyness" && (
                <KeynessLanding onBack={handleBack} genre={selectedGenre} />   
      )}
            {activePage === "clustering" && (
                <ClusteringLanding onBack={handleBack} genre={selectedGenre} />
      )}
            {activePage === "sentiment" && (
                <SentimentLanding onBack={handleBack} genre={selectedGenre} /> 
      )}
            {activePage === "sensorimotor" && (
                <SensorimotorLanding onBack={handleBack} genre={selectedGenre} />
      )}
        </div>
    );
}

export default App;
