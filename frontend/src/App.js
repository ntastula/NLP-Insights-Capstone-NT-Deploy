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

    // --- NEW: handleProceed ---
    const handleProceed = ({ analysisType, genre }) => {
        setSelectedGenre(genre || ""); // set the selected genre (empty for clustering)
        setActivePage(analysisType);   // set the active page to the analysis type
    };

    useUnloadCleanup();

    return (
        <div className="App p-6">
            {activePage === "home" && (
                <HomePage
                    onSelect={setActivePage}
                    selectedGenre={selectedGenre}        
                    onSelectGenre={setSelectedGenre}  
                    onProceed={handleProceed}       // pass it here
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
