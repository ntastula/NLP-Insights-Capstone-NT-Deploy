import React, { useState, useEffect } from "react";
import HomePage from "./Components/HomePage";
import KeynessLanding from "./Components/Keyness/KeynessLanding";
import ClusteringLanding from "./Components/Clustering/ClusteringLanding";
import SentimentLanding from "./Components/Sentiment/SentimentLanding";
import SensorimotorLanding from "./Components/Sensorimotor/SensorimotorLanding";
import useUnloadCleanup from "./Hooks/useUnloadCleanup";
import KeynessWordDetail from "./Components/Keyness/KeynessWordDetail";
import CreativeKeynessResults from "./Components/Keyness/CreativeKeynessResults";

function App() {
    const [activePage, setActivePage] = useState("home");
    const [selectedGenre, setSelectedGenre] = useState("");
    const [wordDetailData, setWordDetailData] = useState(null);
    const handleBack = () => setActivePage("home");
    const [creativeKeynessData, setCreativeKeynessData] = useState(null);
    const [method, setMethod] = useState(""); 
    const [uploadedText, setUploadedText] = useState("");
    const [chartData, setChartData] = useState(null);
    const [summaryLoading, setSummaryLoading] = useState(false);
    const [summary, setSummary] = useState("");
    const [posGroups, setPosGroups] = useState({});
    const [stats, setStats] = useState({});
    const [comparisonMode, setComparisonMode] = useState("");
    const [analysisType, setAnalysisType] = useState("");

    useEffect(() => {
  console.log("Active page changed:", activePage);
}, [activePage]);

    
    const handleProceed = ({ analysisType, genre, comparisonMode }) => {
  console.log("Parent received from HomePage:", { analysisType, genre, comparisonMode });
  setSelectedGenre(genre || null);
  setActivePage(analysisType);
  setComparisonMode(comparisonMode || null);
  setAnalysisType(analysisType);

  console.log("Parent state after update:", {
    selectedGenre: genre,
    comparisonMode,
    analysisType,
  });
};


    const handleKeynessResults = (resultsData) => {
    console.log("Received keyness results:", resultsData);
    setCreativeKeynessData(resultsData.results);
    setMethod(resultsData.method);
    setUploadedText(resultsData.uploadedText);
    setStats(resultsData.stats);
    setActivePage("keyness-results");
    console.log("Parent passing to KeynessLanding:", {
    genre: selectedGenre,
    comparisonMode: comparisonMode,
});
};

    const handleWordDetail = (data) => {
    console.log("Opening word detail for:", data.word);
    if (!creativeKeynessData && data.results) {
        setCreativeKeynessData(data.results);
        setMethod(data.method);
        setUploadedText(data.uploadedText);
        setStats(data.stats || {});
    }
    setWordDetailData(data);
    setActivePage("keyness-word-detail");
};

    const handleBackFromWordDetail = () => {
        console.log("Back from word detail clicked");
        setActivePage("keyness-results"); 
        setTimeout(() => setWordDetailData(null), 0);
    };
    
    useUnloadCleanup();

    return (
        <div className="App p-6">
            {activePage === "home" && (
                <HomePage
                    onSelect={setActivePage}
                    selectedGenre={selectedGenre}        
                    onSelectGenre={setSelectedGenre}  
                    onProceed={handleProceed}
  />
)}
            {activePage === "keyness" && (
                <KeynessLanding 
                    onBack={handleBack} 
                    genre={selectedGenre}
                    onWordDetail={handleWordDetail}
                    posGroups={posGroups}
                    onResults={handleKeynessResults}
                    comparisonMode={comparisonMode} 
                />   
            )}
            {activePage === "keyness-results" && (
                <CreativeKeynessResults 
                    onBackFromWordDetail={handleBackFromWordDetail}
                    onBack={handleBack} 
                    genre={selectedGenre}
                    onWordDetail={handleWordDetail} 
                    results={creativeKeynessData}
                    method={method} 
                    uploadedText={uploadedText} 
                    chartData={chartData}
                    summaryLoading={summaryLoading}
                    summary={summary}
                    posGroups={posGroups}
                    stats={stats}
                />   
            )}
            {activePage === "keyness-word-detail" && wordDetailData && (
                <KeynessWordDetail 
                    {...wordDetailData}
                    onBack={handleBackFromWordDetail}
                />
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