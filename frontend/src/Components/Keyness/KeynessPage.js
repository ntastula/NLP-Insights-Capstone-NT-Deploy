import React, { useState } from "react";
import CreativeKeynessResults from "./CreativeKeynessResults";
import KeynessWordDetail from "./KeynessWordDetail";

const KeynessPage = ({ results, stats, method, uploadedText, genre }) => {
  const [selectedWordDetail, setSelectedWordDetail] = useState(null);

  const handleWordDetail = (detail) => {
    setSelectedWordDetail(detail);
  };

  const handleBack = () => {
    setSelectedWordDetail(null);
  };

  return (
    <div>
      {selectedWordDetail ? (
        <KeynessWordDetail
          word={selectedWordDetail.word}
          wordData={selectedWordDetail.wordData}
          uploadedText={selectedWordDetail.uploadedText}
          method={selectedWordDetail.method}
          results={selectedWordDetail.results}
          onBack={handleBack}
        />
      ) : (
        <CreativeKeynessResults
          results={results}
          stats={stats}
          method={method}
          uploadedText={uploadedText}
          genre={genre}
          onWordDetail={handleWordDetail}
        />
      )}
    </div>
  );
};

export default KeynessPage;
