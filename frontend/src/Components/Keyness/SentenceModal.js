import React from "react";
import "./SentenceModal.css";

const SentenceModal = ({ word, sentences, onClose }) => {
  if (!word) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">
          Sentences containing: <span className="highlight">{word}</span>
        </h3>

        <div className="sentence-list">
          {sentences.length > 0 ? (
            sentences.map((s, idx) => {
              const regex = new RegExp(`(${word})`, "gi");
              const parts = s.split(regex);
              return (
                <p key={idx} className="sentence">
                  {parts.map((part, i) =>
                    part.toLowerCase() === word.toLowerCase() ? (
                      <span key={i} className="highlight">{part}</span>
                    ) : (
                      part
                    )
                  )}
                </p>
              );
            })
          ) : (
            <p className="no-results">No sentences found.</p>
          )}
        </div>

        <button className="close-btn" onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default SentenceModal;
