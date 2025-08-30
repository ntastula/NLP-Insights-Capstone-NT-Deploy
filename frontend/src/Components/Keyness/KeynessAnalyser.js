import React, { useState, useRef } from "react";
import ResultsTable from "./ResultsTable";
import KeynessResultsGrid from "./KeynessResultsGrid";
import Charts from "./Charts";
import ResultsSummary from "./ResultsSummary";

const KeynessAnalyser = ({ uploadedText, uploadedPreview, corpusPreview, method, onBack }) => {
  const [comparisonResults, setComparisonResults] = useState([]);
  const [stats, setStats] = useState({ uploaded_total: 0, sample_total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisDone, setAnalysisDone] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState("nltk");
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // Rate limiting state
  const [lastRequestTime, setLastRequestTime] = useState(0);
  const [requestCount, setRequestCount] = useState(0);
  const REQUEST_LIMIT = 10; // requests per minute
  const RATE_LIMIT_WINDOW = 60000; // 1 minute in ms
  
  const abortControllerRef = useRef(null);

  // Input validation functions
  const validateTextInput = (text) => {
    if (!text || typeof text !== 'string') {
      throw new Error('Invalid text input: Text must be a non-empty string');
    }
    
    if (text.trim().length === 0) {
      throw new Error('Text cannot be empty or contain only whitespace');
    }
    
    if (text.length > 1000000) { // 1MB text limit
      throw new Error('Text is too long. Maximum length is 1,000,000 characters');
    }
    
    if (text.length < 50) {
      throw new Error('Text is too short. Minimum length is 50 characters for meaningful analysis');
    }
    
    // Check for potential script injection
    const scriptRegex = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi;
    if (scriptRegex.test(text)) {
      throw new Error('Text contains potentially malicious content');
    }
    
    return true;
  };

  const validateMethod = (method) => {
    const allowedMethods = ['nltk', 'sklearn', 'gensim', 'spacy'];
    if (!allowedMethods.includes(method.toLowerCase())) {
      throw new Error(`Invalid analysis method. Allowed methods: ${allowedMethods.join(', ')}`);
    }
    return method.toLowerCase();
  };

  // Rate limiting check
  const checkRateLimit = () => {
    const now = Date.now();
    
    // Reset counter if window has passed
    if (now - lastRequestTime > RATE_LIMIT_WINDOW) {
      setRequestCount(1);
      setLastRequestTime(now);
      return true;
    }
    
    // Check if we've exceeded the limit
    if (requestCount >= REQUEST_LIMIT) {
      throw new Error(`Rate limit exceeded. Please wait before making another request.`);
    }
    
    setRequestCount(prev => prev + 1);
    return true;
  };

  // Secure API call with timeout and abort capability
  const makeSecureAPICall = async (payload, method) => {
    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();
    
    const timeoutId = setTimeout(() => {
      abortControllerRef.current?.abort();
    }, 30000); // 30 second timeout

    try {
      const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
      
      const response = await fetch("http://localhost:8000/api/analyse-keyness/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfToken,
        },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal,
        mode: 'cors',
      });

      clearTimeout(timeoutId);

      // Check if response is ok
      if (!response.ok) {
        if (response.status === 413) {
          throw new Error('File too large for processing');
        } else if (response.status === 429) {
          throw new Error('Too many requests. Please wait before trying again');
        } else if (response.status === 401) {
          throw new Error('Authentication required. Please log in');
        } else if (response.status === 403) {
          throw new Error('Access denied. You do not have permission for this action');
        } else {
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
      }

      // Validate response content type
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Invalid response format from server');
      }

      const data = await response.json();

      // Validate response structure
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid response data from server');
      }

      return data;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new Error('Request timeout. Please try again');
      }
      
      // Re-throw validation and network errors
      throw error;
    }
  };

  const performAnalysis = async (method) => {
    try {
      // Input validation
      validateTextInput(uploadedText);
      const validatedMethod = validateMethod(method);
      
      // Rate limiting check
      checkRateLimit();

      setLoading(true);
      setError("");
      setAnalysisDone(false);
      setSelectedMethod(validatedMethod);
      setUploadProgress(0);

      console.log("Perform analysis clicked. Method:", validatedMethod);

      // Simulate upload progress (in a real app, you'd get this from the upload)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 100);

      const payload = {
        uploaded_text: uploadedText.trim(),
        method: validatedMethod,
        // Add timestamp for request tracking
        timestamp: Date.now(),
        // Add text hash for integrity checking (simple example)
        text_hash: btoa(uploadedText.slice(0, 100)).slice(0, 16)
      };

      const data = await makeSecureAPICall(payload, validatedMethod);

      clearInterval(progressInterval);
      setUploadProgress(100);

      // Validate response data structure
      if (data.error) {
        throw new Error(data.error);
      }

      if (!data.results) {
        throw new Error('Invalid response: missing results data');
      }

      console.log("Received data:", data);

      // Sanitize and validate results before setting state
      const sanitizedResults = Array.isArray(data.results.results || data.results) 
        ? (data.results.results || data.results).map(result => ({
          ...result,
          // Ensure numeric values are actually numbers
          score: typeof result.score === 'number' ? result.score : parseFloat(result.score) || 0,
          frequency: typeof result.frequency === 'number' ? result.frequency : parseInt(result.frequency) || 0
        }))
        : [];

      // Update state with validated results
      setComparisonResults(sanitizedResults);
      setStats({
        uploadedTotal: typeof data.uploaded_total === 'number' 
          ? data.uploaded_total 
          : uploadedText.split(/\s+/).length,
        corpusTotal: typeof data.corpus_total === 'number' 
          ? data.corpus_total 
          : 0
      });

      setAnalysisDone(true);

    } catch (err) {
      console.error("Analysis error:", err);
      setError(`Analysis failed: ${err.message}`);
      
      // Reset progress on error
      setUploadProgress(0);
    } finally {
      setLoading(false);
      // Clean up abort controller
      abortControllerRef.current = null;
    }
  };

  // Cancel ongoing request
  const cancelAnalysis = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
      setError("Analysis cancelled by user");
      setUploadProgress(0);
    }
  };

  return (
    <div className="mb-6">
      <button
        onClick={onBack}
        className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
      >
        ← Back
      </button>

      {/* Analysis Controls */}
      <div className="text-center mb-6 flex justify-center gap-4 flex-wrap">
        <button
          onClick={() => performAnalysis("NLTK")}
          disabled={loading || !uploadedText}
          className="btn disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Analyse with NLTK
        </button>

        <button
          onClick={() => performAnalysis("sklearn")}
          disabled={loading || !uploadedText}
          className="btn disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Analyse with Scikit-Learn
        </button>

        <button
          onClick={() => performAnalysis("gensim")}
          disabled={loading || !uploadedText}
          className="btn disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Analyse with Gensim
        </button>

        <button
          onClick={() => performAnalysis("spaCy")}
          disabled={loading || !uploadedText}
          className="btn disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Analyse with spaCy
        </button>

        {loading && (
          <button
            onClick={cancelAnalysis}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress Indicator */}
      {loading && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>Analyzing text...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Validation Info */}
      {uploadedText && (
        <div className="mb-4 text-sm text-gray-600">
          Text length: {uploadedText.length.toLocaleString()} characters 
          ({Math.round(uploadedText.split(/\s+/).length).toLocaleString()} words)
        </div>
      )}

      {analysisDone && (
        <>
          {/* Results Summary */}
          <ResultsSummary
            stats={stats}
            selectedMethod={selectedMethod}
            comparisonResults={comparisonResults}
          />

          {/* Significant Keywords Grid */}
          <KeynessResultsGrid results={comparisonResults.slice(0, 20)} method={selectedMethod} />

          {/* Charts */}
          <Charts results={comparisonResults.results ?? comparisonResults} method={selectedMethod} />

          {/* Full Results Table */}
          <ResultsTable results={comparisonResults} method={selectedMethod} />
        </>
      )}
    </div>
  );
};

export default KeynessAnalyser;
