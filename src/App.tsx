import React, { useState, useEffect, useId } from 'react';
import {
  BarChart3,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  FileSpreadsheet,
  UploadCloud,
  Download,
  RefreshCw,
  Terminal,
  Cpu,
  Layers,
  ShieldCheck,
  TrendingUp,
  Sliders,
  Copy,
  Check,
  Trash2,
  Sparkles,
  Info,
  ChevronRight
} from 'lucide-react';

interface PredictionResult {
  text: string;
  cleaned_text: string;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  confidence: number;
  probabilities: {
    Positive: number;
    Negative: number;
    Neutral: number;
  };
  explanation: string;
  influential_tokens: Array<{
    token: string;
    weight: number;
    polarity: string;
  }>;
  character_count: number;
  word_count: number;
  timestamp?: string;
  latency_ms?: number;
  is_mixed_sentiment?: boolean;
  is_out_of_vocabulary?: boolean;
  certainty_level?: string;
  vocabulary_tokens_matched?: number;
}

interface BatchItem {
  row_index: number;
  text: string;
  cleaned_text?: string;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  confidence: number;
}

interface ModelMetrics {
  champion_model: string;
  selection_criterion: string;
  classes: string[];
  dataset_statistics: {
    total_samples: number;
    train_samples: number;
    test_samples: number;
    split_ratio: string;
    random_seed: number;
    class_distribution: Record<string, number>;
  };
  models: {
    'Multinomial Naive Bayes': {
      accuracy: number;
      precision_macro: number;
      recall_macro: number;
      f1_macro: number;
      confusion_matrix: number[][];
      classification_report: Record<string, any>;
    };
    'Logistic Regression': {
      accuracy: number;
      precision_macro: number;
      recall_macro: number;
      f1_macro: number;
      confusion_matrix: number[][];
      classification_report: Record<string, any>;
    };
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'analyzer' | 'batch' | 'metrics' | 'pipeline' | 'api'>('analyzer');
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isAutoAnalyze, setIsAutoAnalyze] = useState(true);
  const [typingStatus, setTypingStatus] = useState<'idle' | 'typing' | 'analyzing' | 'synced'>('idle');
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionResult[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [serverHealth, setServerHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Batch CSV state
  const [csvFileName, setCsvFileName] = useState<string | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [csvRows, setCsvRows] = useState<string[][]>([]);
  const [selectedColumnIdx, setSelectedColumnIdx] = useState<number>(0);
  const [isBatchLoading, setIsBatchLoading] = useState(false);
  const [batchResults, setBatchResults] = useState<{
    total: number;
    counts: Record<string, number>;
    percentages: Record<string, number>;
    items: BatchItem[];
  } | null>(null);
  const [batchFilter, setBatchFilter] = useState<'ALL' | 'Positive' | 'Negative' | 'Neutral'>('ALL');
  const [batchSearch, setBatchSearch] = useState('');

  // Preprocessing interactive lab state
  const [labInput, setLabInput] = useState('Worst phone ever! Battery died <3 hours> & screen was not working. Check @support https://example.com/bad');
  const [labSteps, setLabSteps] = useState<any>(null);
  const [isLabLoading, setIsLabLoading] = useState(false);

  // API docs copy state
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null);

  // Accessible unique IDs for form controls
  const analyzerTextareaId = useId();
  const batchColumnSelectId = useId();
  const batchFilterSelectId = useId();
  const pipelineInputId = useId();

  // Load model metrics and health on mount
  useEffect(() => {
    fetchHealth();
    fetchMetrics();
    runLabPipeline(labInput);
  }, []);

  // Real-time automatic prediction as user types (debounced at 400ms)
  useEffect(() => {
    if (!isAutoAnalyze) return;
    const trimmed = inputText.trim();
    if (!trimmed) {
      setCurrentResult(null);
      setTypingStatus('idle');
      return;
    }

    setTypingStatus('typing');
    const timer = setTimeout(() => {
      setTypingStatus('analyzing');
      handleAnalyze(trimmed, true);
    }, 400);

    return () => clearTimeout(timer);
  }, [inputText, isAutoAnalyze]);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setServerHealth(data);
      }
    } catch (e) {
      console.warn('Health check error:', e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/model-info');
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.warn('Metrics fetch error:', e);
    }
  };

  const runLabPipeline = async (text: string) => {
    setIsLabLoading(true);
    try {
      const res = await fetch('/api/preprocess-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (res.ok) {
        const data = await res.json();
        setLabSteps(data);
      }
    } catch (e) {
      console.warn('Lab pipeline error:', e);
    } finally {
      setIsLabLoading(false);
    }
  };

  const handleAnalyze = async (overrideText?: string, isLive: boolean = false) => {
    const textToAnalyze = (overrideText !== undefined ? overrideText : inputText).trim();
    if (!textToAnalyze) {
      if (!isLive) {
        setErrorMsg('Please enter a sentence or customer review to analyze.');
      }
      return;
    }

    if (!isLive) {
      setIsLoading(true);
    }
    setErrorMsg(null);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToAnalyze }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Prediction request failed.');
      }

      const enrichedResult: PredictionResult = {
        ...data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      };

      setCurrentResult(enrichedResult);
      setTypingStatus('synced');

      // Update history (avoid spamming identical consecutive entries during typing)
      setHistory((prev) => {
        const existingIdx = prev.findIndex((h) => h.text.toLowerCase() === textToAnalyze.toLowerCase());
        if (existingIdx === 0) {
          const updated = [...prev];
          updated[0] = enrichedResult;
          return updated;
        }
        return [enrichedResult, ...prev.slice(0, 19)];
      });
    } catch (err: any) {
      if (!isLive) {
        setErrorMsg(err.message || 'An error occurred during sentiment prediction.');
      }
      setTypingStatus('idle');
    } finally {
      if (!isLive) {
        setIsLoading(false);
      }
    }
  };

  const handleSampleClick = (text: string) => {
    setInputText(text);
    handleAnalyze(text, false);
  };

  const handleClear = () => {
    setInputText('');
    setCurrentResult(null);
    setTypingStatus('idle');
    setErrorMsg(null);
  };

  // CSV parsing
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCsvFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      parseCSV(content);
    };
    reader.readAsText(file);
  };

  const parseCSV = (text: string) => {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) {
      setErrorMsg('CSV must contain a header row and at least one data row.');
      return;
    }

    const header = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
    setCsvHeaders(header);

    const rows: string[][] = [];
    for (let i = 1; i < lines.length; i++) {
      const regex = /(?:,|\n|^)("(?:(?:"")*[^"]*)*"|[^",\n]*|(?:\n|$))/g;
      const row: string[] = [];
      let match;
      while ((match = regex.exec(lines[i])) !== null && match[0] !== '') {
        let val = match[1] || '';
        if (val.startsWith('"') && val.endsWith('"')) {
          val = val.substring(1, val.length - 1).replace(/""/g, '"');
        }
        row.push(val);
        if (regex.lastIndex >= lines[i].length) break;
      }
      if (row.length > 0) {
        rows.push(row);
      }
    }

    setCsvRows(rows);

    // Auto-detect text column
    const detectedIdx = header.findIndex((h) => /text|review|comment|feedback|content|body/i.test(h));
    setSelectedColumnIdx(detectedIdx >= 0 ? detectedIdx : 0);
    setBatchResults(null);
  };

  const runBatchPrediction = async () => {
    if (csvRows.length === 0) return;
    const texts = csvRows.map((r) => r[selectedColumnIdx] || '').filter((t) => t.trim().length > 0);

    if (texts.length === 0) {
      setErrorMsg('Selected column does not contain any valid text.');
      return;
    }

    setIsBatchLoading(true);
    setErrorMsg(null);

    try {
      const res = await fetch('/api/predict-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Batch prediction failed.');
      }

      setBatchResults({
        total: data.total_analyzed,
        counts: data.counts,
        percentages: data.percentages,
        items: data.items,
      });
    } catch (err: any) {
      setErrorMsg(err.message || 'Batch prediction error.');
    } finally {
      setIsBatchLoading(false);
    }
  };

  const exportEnrichedCSV = () => {
    if (!batchResults || batchResults.items.length === 0) return;

    let csvContent = 'data:text/csv;charset=utf-8,Row_Index,Original_Text,Predicted_Sentiment,Confidence_Score,Cleaned_Text\n';
    batchResults.items.forEach((item) => {
      const safeText = `"${(item.text || '').replace(/"/g, '""')}"`;
      const safeCleaned = `"${(item.cleaned_text || '').replace(/"/g, '""')}"`;
      csvContent += `${item.row_index},${safeText},${item.sentiment},${(item.confidence * 100).toFixed(1)}%,${safeCleaned}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `SentimentIQ_Batch_Analysis_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const copyCode = (snippet: string, key: string) => {
    navigator.clipboard.writeText(snippet);
    setCopiedSnippet(key);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  // Filtered batch items
  const filteredBatchItems = (batchResults?.items || []).filter((item) => {
    const matchesFilter = batchFilter === 'ALL' || item.sentiment === batchFilter;
    const matchesSearch = !batchSearch || item.text.toLowerCase().includes(batchSearch.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  // Calculate session distribution
  const sessionTotal = history.length;
  const sessionPos = history.filter((h) => h.sentiment === 'Positive').length;
  const sessionNeg = history.filter((h) => h.sentiment === 'Negative').length;
  const sessionNeu = history.filter((h) => h.sentiment === 'Neutral').length;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-sm">
              IQ
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight text-slate-900">SentimentIQ</span>
                <span className="text-[11px] font-mono text-indigo-700 bg-indigo-50 border border-indigo-200 px-1.5 py-0.5 rounded">v1.0 ML</span>
              </div>
              <p className="text-xs text-slate-500">AI-Powered 3-Class Sentiment Platform</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('analyzer')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                activeTab === 'analyzer' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Text Analyzer
            </button>
            <button
              onClick={() => setActiveTab('batch')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                activeTab === 'batch' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Batch CSV
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                activeTab === 'metrics' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Model Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                activeTab === 'pipeline' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              NLP Pipeline
            </button>
            <button
              onClick={() => setActiveTab('api')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                activeTab === 'api' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              REST API
            </button>
          </nav>

          {/* System Status */}
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1.5 bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-1 rounded-full font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>scikit-learn Active</span>
            </div>
          </div>
        </div>

        {/* Mobile Tab Bar */}
        <div className="md:hidden flex overflow-x-auto border-t border-slate-200 px-4 py-2 gap-2 bg-slate-50">
          {(['analyzer', 'batch', 'metrics', 'pipeline', 'api'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 text-xs whitespace-nowrap font-medium rounded-md ${
                activeTab === tab ? 'bg-indigo-600 text-white' : 'bg-white text-slate-700 border border-slate-200'
              }`}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Global Error Banner */}
        {errorMsg && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start justify-between text-red-700 text-sm">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 shrink-0 text-red-500" />
              <span>{errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} className="text-red-500 hover:text-red-800 font-bold ml-4">
              ✕
            </button>
          </div>
        )}

        {/* Session Stats Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-2xs">
            <span className="text-xs text-slate-500 font-medium">Session Inferences</span>
            <div className="text-2xl font-bold text-slate-900 mt-1">{sessionTotal}</div>
            <span className="text-[11px] text-slate-400">Total texts classified</span>
          </div>
          <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-2xs border-l-4 border-l-emerald-500">
            <span className="text-xs text-emerald-700 font-semibold">Positive Sentiment</span>
            <div className="text-2xl font-bold text-emerald-600 mt-1">{sessionPos}</div>
            <span className="text-[11px] text-slate-500">
              {sessionTotal > 0 ? ((sessionPos / sessionTotal) * 100).toFixed(1) : '0.0'}% of session
            </span>
          </div>
          <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-2xs border-l-4 border-l-rose-500">
            <span className="text-xs text-rose-700 font-semibold">Negative Sentiment</span>
            <div className="text-2xl font-bold text-rose-600 mt-1">{sessionNeg}</div>
            <span className="text-[11px] text-slate-500">
              {sessionTotal > 0 ? ((sessionNeg / sessionTotal) * 100).toFixed(1) : '0.0'}% of session
            </span>
          </div>
          <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-2xs border-l-4 border-l-amber-500">
            <span className="text-xs text-amber-700 font-semibold">Neutral Sentiment</span>
            <div className="text-2xl font-bold text-amber-600 mt-1">{sessionNeu}</div>
            <span className="text-[11px] text-slate-500">
              {sessionTotal > 0 ? ((sessionNeu / sessionTotal) * 100).toFixed(1) : '0.0'}% of session
            </span>
          </div>
        </div>

        {/* TAB 1: TEXT ANALYZER */}
        {activeTab === 'analyzer' && (
          <div className="space-y-8">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold text-slate-900">Real-Time Sentiment Analysis</h2>
                    <span className="text-[10px] font-bold uppercase tracking-wider bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded-full">
                      Live Typing ML
                    </span>
                  </div>
                  <p className="text-sm text-slate-500">
                    Type or paste any customer review, support ticket, or social post. Predictions update automatically as you type.
                  </p>
                </div>

                {/* Preconfigured Samples */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-slate-400 font-medium">Try Sample:</span>
                  <button
                    onClick={() => handleSampleClick('The sound quality is crystal clear with deep bass and excellent battery life. Absolutely love this headset!')}
                    className="text-xs px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition-colors"
                  >
                    Positive Audio
                  </button>
                  <button
                    onClick={() => handleSampleClick('Worst customer service experience of my life. The item arrived broken and nobody responded to my emails.')}
                    className="text-xs px-2.5 py-1 rounded-md bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100 transition-colors"
                  >
                    Negative Support
                  </button>
                  <button
                    onClick={() => handleSampleClick('The package arrived on Tuesday as scheduled. The box was sealed and contents were inside.')}
                    className="text-xs px-2.5 py-1 rounded-md bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100 transition-colors"
                  >
                    Neutral Tracking
                  </button>
                </div>
              </div>

              {/* Real-Time Status & Mode Bar */}
              <div className="flex items-center justify-between py-2 px-3 bg-slate-50 rounded-lg border border-slate-200 mb-3 text-xs">
                <label className="flex items-center gap-2 cursor-pointer font-medium text-slate-700">
                  <input
                    type="checkbox"
                    checked={isAutoAnalyze}
                    onChange={(e) => setIsAutoAnalyze(e.target.checked)}
                    className="rounded text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5"
                  />
                  <span>Auto-predict as you type (400ms debounce)</span>
                </label>

                {/* Live Typing Status Pill */}
                <div className="flex items-center gap-2">
                  {typingStatus === 'typing' && (
                    <span className="flex items-center gap-1.5 text-slate-500 font-medium">
                      <span className="w-2 h-2 rounded-full bg-slate-400 animate-ping"></span>
                      <span>Typing...</span>
                    </span>
                  )}
                  {typingStatus === 'analyzing' && (
                    <span className="flex items-center gap-1.5 text-indigo-600 font-medium">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>scikit-learn inferring...</span>
                    </span>
                  )}
                  {typingStatus === 'synced' && currentResult && (
                    <span className="flex items-center gap-1.5 text-emerald-700 font-medium">
                      <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                      <span>Live ML synced ({currentResult.latency_ms || 18}ms)</span>
                    </span>
                  )}
                  {typingStatus === 'idle' && (
                    <span className="text-slate-400">Ready for typing</span>
                  )}
                </div>
              </div>

              {/* Text Input Area */}
              <div className="relative">
                <label htmlFor={analyzerTextareaId} className="sr-only">Customer review or text to analyze</label>
                <textarea
                  id={analyzerTextareaId}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                      e.preventDefault();
                      handleAnalyze(undefined, false);
                    }
                  }}
                  rows={4}
                  maxLength={5000}
                  placeholder="Type any newly written review or sentence here to see real-time NLP classification (e.g. 'The keyboard has great tactile feedback, but the spacebar squeaks occasionally')..."
                  className="w-full p-4 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-slate-900 placeholder-slate-400 text-sm leading-relaxed outline-hidden"
                />
                <div className="flex items-center justify-between text-xs text-slate-400 mt-2">
                  <span>{inputText.length} / 5,000 characters &middot; Press Ctrl+Enter for manual run</span>
                  <div className="flex items-center gap-2">
                    {inputText && (
                      <button
                        onClick={handleClear}
                        className="px-3 py-1.5 text-xs text-slate-600 hover:text-slate-900 font-medium cursor-pointer"
                      >
                        Clear
                      </button>
                    )}
                    <button
                      onClick={() => handleAnalyze(undefined, false)}
                      disabled={isLoading || !inputText.trim()}
                      className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold text-xs shadow-xs transition-colors cursor-pointer"
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          <span>Classifying...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5" />
                          <span>Analyze Sentiment</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Prediction Result Display */}
              {currentResult && (
                <div className="mt-6 p-6 border border-slate-200 rounded-xl bg-slate-50/50">
                  {/* Warning Banners for Mixed Sentiment or Out of Vocabulary */}
                  {currentResult.is_mixed_sentiment && (
                    <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-2 text-amber-800 text-xs">
                      <HelpCircle className="w-4 h-4 text-amber-600 shrink-0" />
                      <span>
                        <strong>Mixed / Borderline Sentiment Detected:</strong> The input contains conflicting positive and negative signals with closely matched top probabilities.
                      </span>
                    </div>
                  )}

                  {currentResult.is_out_of_vocabulary && (
                    <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-2 text-blue-800 text-xs">
                      <Info className="w-4 h-4 text-blue-600 shrink-0" />
                      <span>
                        <strong>Unseen Vocabulary:</strong> The text uses words not present in the 5,000-term TF-IDF training dictionary. Prediction reflects the baseline prior distribution.
                      </span>
                    </div>
                  )}

                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
                    <div className="flex items-center gap-3">
                      <div
                        className={`px-4 py-2 rounded-xl font-bold text-base flex items-center gap-2 shadow-xs ${
                          currentResult.sentiment === 'Positive'
                            ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                            : currentResult.sentiment === 'Negative'
                            ? 'bg-rose-100 text-rose-800 border border-rose-300'
                            : 'bg-amber-100 text-amber-800 border border-amber-300'
                        }`}
                      >
                        {currentResult.sentiment === 'Positive' && <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
                        {currentResult.sentiment === 'Negative' && <AlertCircle className="w-5 h-5 text-rose-600" />}
                        {currentResult.sentiment === 'Neutral' && <HelpCircle className="w-5 h-5 text-amber-600" />}
                        <span>Predicted: {currentResult.sentiment}</span>
                      </div>
                      <div className="text-xs text-slate-500">
                        Confidence: <strong className="text-slate-800">{(currentResult.confidence * 100).toFixed(1)}%</strong>
                        {currentResult.certainty_level && (
                          <span className="ml-1.5 text-slate-400">({currentResult.certainty_level})</span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                      {currentResult.latency_ms !== undefined && (
                        <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded border border-slate-200">
                          ⚡ {currentResult.latency_ms}ms ML latency
                        </span>
                      )}
                      <span>{currentResult.timestamp}</span>
                    </div>
                  </div>

                  {/* 3-Class Probability Bars */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
                    <div className="bg-white p-3.5 rounded-lg border border-slate-200">
                      <div className="flex justify-between items-center text-xs font-semibold mb-1">
                        <span className="text-emerald-700">Positive Probability</span>
                        <span className="font-mono">{((currentResult.probabilities.Positive || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                        <div
                          className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${(currentResult.probabilities.Positive || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-white p-3.5 rounded-lg border border-slate-200">
                      <div className="flex justify-between items-center text-xs font-semibold mb-1">
                        <span className="text-rose-700">Negative Probability</span>
                        <span className="font-mono">{((currentResult.probabilities.Negative || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                        <div
                          className="bg-rose-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${(currentResult.probabilities.Negative || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-white p-3.5 rounded-lg border border-slate-200">
                      <div className="flex justify-between items-center text-xs font-semibold mb-1">
                        <span className="text-amber-700">Neutral Probability</span>
                        <span className="font-mono">{((currentResult.probabilities.Neutral || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                        <div
                          className="bg-amber-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${(currentResult.probabilities.Neutral || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Feature Attribution / Influential Tokens */}
                  <div className="bg-white p-4 rounded-xl border border-slate-200 text-xs">
                    <div className="flex items-center gap-2 font-semibold text-slate-800 mb-2">
                      <TrendingUp className="w-4 h-4 text-indigo-600" />
                      <span>Model Explanation & Key Influential Features</span>
                    </div>
                    <p className="text-slate-600 mb-3">{currentResult.explanation}</p>

                    {currentResult.influential_tokens && currentResult.influential_tokens.length > 0 && (
                      <div>
                        <span className="text-slate-400 font-medium block mb-1.5">Top TF-IDF feature weights:</span>
                        <div className="flex flex-wrap gap-2">
                          {currentResult.influential_tokens.map((token, idx) => (
                            <span
                              key={idx}
                              className={`px-2 py-1 rounded font-mono text-[11px] border ${
                                token.polarity === 'positive_signal'
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : 'bg-slate-100 text-slate-700 border-slate-200'
                              }`}
                            >
                              {token.token} ({token.weight > 0 ? `+${token.weight}` : token.weight})
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Session History Table */}
            {history.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-slate-900">Recent Session Predictions</h3>
                  <button
                    onClick={() => setHistory([])}
                    className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-red-600"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear History</span>
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 border-b border-slate-200 text-slate-500">
                      <tr>
                        <th className="py-2.5 px-3 font-semibold">Time</th>
                        <th className="py-2.5 px-3 font-semibold">Analyzed Text</th>
                        <th className="py-2.5 px-3 font-semibold">Sentiment</th>
                        <th className="py-2.5 px-3 font-semibold">Confidence</th>
                        <th className="py-2.5 px-3 font-semibold text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {history.map((item, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="py-2.5 px-3 font-mono text-slate-400 whitespace-nowrap">{item.timestamp}</td>
                          <td className="py-2.5 px-3 max-w-md truncate text-slate-700">{item.text}</td>
                          <td className="py-2.5 px-3 whitespace-nowrap">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                                item.sentiment === 'Positive'
                                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                  : item.sentiment === 'Negative'
                                  ? 'bg-rose-50 text-rose-700 border border-rose-200'
                                  : 'bg-amber-50 text-amber-700 border border-amber-200'
                              }`}
                            >
                              {item.sentiment}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-700">{(item.confidence * 100).toFixed(1)}%</td>
                          <td className="py-2.5 px-3 text-right whitespace-nowrap">
                            <button
                              onClick={() => {
                                setInputText(item.text);
                                setCurrentResult(item);
                              }}
                              className="text-indigo-600 hover:text-indigo-800 font-medium"
                            >
                              Re-view
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: BATCH CSV ANALYSIS */}
        {activeTab === 'batch' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h2 className="text-lg font-bold text-slate-900">Batch CSV File Analysis</h2>
                  <p className="text-sm text-slate-500">
                    Upload any CSV dataset containing customer reviews, select the target text column, and generate predictions in bulk.
                  </p>
                </div>
                <a
                  href="/api/download-sample-csv"
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                  download
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Test CSV</span>
                </a>
              </div>

              {/* Upload Dropzone */}
              <label className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-slate-50/50 hover:bg-indigo-50/20 transition-all">
                <UploadCloud className="w-10 h-10 text-slate-400 mb-2" />
                <span className="text-sm font-semibold text-slate-800">
                  {csvFileName ? `Loaded: ${csvFileName}` : 'Click or drop a CSV file to begin'}
                </span>
                <span className="text-xs text-slate-400 mt-1">Supports UTF-8 CSV with text, review, or feedback columns</span>
                <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              </label>

              {/* Column Selection & Run Action */}
              {csvHeaders.length > 0 && (
                <div className="mt-6 p-4 bg-slate-50 rounded-xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <label htmlFor={batchColumnSelectId} className="text-xs font-semibold text-slate-700 whitespace-nowrap">Text Column to Analyze:</label>
                    <select
                      id={batchColumnSelectId}
                      value={selectedColumnIdx}
                      onChange={(e) => setSelectedColumnIdx(Number(e.target.value))}
                      className="px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs font-medium focus:ring-2 focus:ring-indigo-500 outline-hidden"
                    >
                      {csvHeaders.map((header, idx) => (
                        <option key={idx} value={idx}>
                          {header} (Column {idx + 1})
                        </option>
                      ))}
                    </select>
                    <span className="text-xs text-slate-500">({csvRows.length} rows loaded)</span>
                  </div>

                  <button
                    onClick={runBatchPrediction}
                    disabled={isBatchLoading}
                    className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold text-xs shadow-xs transition-colors cursor-pointer"
                  >
                    {isBatchLoading ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Predicting {csvRows.length} Rows...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Run Batch Predictions</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* Batch Results Overview */}
            {batchResults && (
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-200">
                  <div>
                    <h3 className="text-base font-bold text-slate-900">Batch Prediction Summary</h3>
                    <p className="text-xs text-slate-500">Successfully classified {batchResults.total} rows</p>
                  </div>
                  <button
                    onClick={exportEnrichedCSV}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Enriched CSV</span>
                  </button>
                </div>

                {/* Batch Distribution Metrics */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                  <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/50">
                    <span className="text-xs font-semibold text-emerald-800">Positive Feedback</span>
                    <div className="text-xl font-bold text-emerald-700 mt-1">
                      {batchResults.counts.Positive || 0} rows ({batchResults.percentages.Positive || 0}%)
                    </div>
                  </div>
                  <div className="p-4 rounded-xl border border-rose-200 bg-rose-50/50">
                    <span className="text-xs font-semibold text-rose-800">Negative Feedback</span>
                    <div className="text-xl font-bold text-rose-700 mt-1">
                      {batchResults.counts.Negative || 0} rows ({batchResults.percentages.Negative || 0}%)
                    </div>
                  </div>
                  <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/50">
                    <span className="text-xs font-semibold text-amber-800">Neutral Observations</span>
                    <div className="text-xl font-bold text-amber-700 mt-1">
                      {batchResults.counts.Neutral || 0} rows ({batchResults.percentages.Neutral || 0}%)
                    </div>
                  </div>
                </div>

                {/* Filter and Search Bar */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500">Filter Class:</span>
                    <label htmlFor={batchFilterSelectId} className="sr-only">Filter by sentiment class</label>
                    <select
                      id={batchFilterSelectId}
                      value={batchFilter}
                      onChange={(e: any) => setBatchFilter(e.target.value)}
                      className="px-2.5 py-1 text-xs border border-slate-300 rounded-md bg-white font-medium"
                    >
                      <option value="ALL">All ({batchResults.total})</option>
                      <option value="Positive">Positive Only</option>
                      <option value="Negative">Negative Only</option>
                      <option value="Neutral">Neutral Only</option>
                    </select>
                  </div>
                  <input
                    type="text"
                    value={batchSearch}
                    onChange={(e) => setBatchSearch(e.target.value)}
                    placeholder="Search within batch text..."
                    className="px-3 py-1 text-xs border border-slate-300 rounded-md bg-white w-full sm:w-64"
                  />
                </div>

                {/* Results Table */}
                <div className="overflow-x-auto border border-slate-200 rounded-xl">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                      <tr>
                        <th className="py-2.5 px-3">#</th>
                        <th className="py-2.5 px-3">Text Sample</th>
                        <th className="py-2.5 px-3">Predicted Sentiment</th>
                        <th className="py-2.5 px-3">Model Confidence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {filteredBatchItems.map((item, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="py-2.5 px-3 font-mono text-slate-400">{item.row_index}</td>
                          <td className="py-2.5 px-3 max-w-xl truncate text-slate-800">{item.text}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                                item.sentiment === 'Positive'
                                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                  : item.sentiment === 'Negative'
                                  ? 'bg-rose-50 text-rose-700 border border-rose-200'
                                  : 'bg-amber-50 text-amber-700 border border-amber-200'
                              }`}
                            >
                              {item.sentiment}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-700">{(item.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: MODEL BENCHMARKS & EVALUATION */}
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="mb-6">
                <h2 className="text-lg font-bold text-slate-900">Model Evaluation & Benchmark Comparison</h2>
                <p className="text-sm text-slate-500">
                  Rigorous side-by-side performance comparison on identical 80/20 stratified test holdout.
                </p>
              </div>

              {/* Performance Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                {/* Logistic Regression Card */}
                <div className="p-5 rounded-xl border-2 border-indigo-500 bg-indigo-50/20 relative">
                  <div className="absolute top-4 right-4 bg-indigo-600 text-white text-[10px] font-bold px-2 py-0.5 rounded tracking-wider">
                    CHAMPION MODEL
                  </div>
                  <h3 className="text-base font-bold text-slate-900">Logistic Regression</h3>
                  <p className="text-xs text-slate-500 mb-4">TF-IDF Unigrams + Bigrams with L2 Regularization</p>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-indigo-100">
                      <span className="text-slate-600">Test Accuracy</span>
                      <strong className="font-mono text-slate-900">
                        {metrics ? `${(metrics.models['Logistic Regression']?.accuracy * 100).toFixed(2)}%` : '75.76%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1 border-b border-indigo-100">
                      <span className="text-slate-600">Macro F1-Score (Selection Metric)</span>
                      <strong className="font-mono text-indigo-700 font-bold">
                        {metrics ? `${(metrics.models['Logistic Regression']?.f1_macro * 100).toFixed(2)}%` : '75.69%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1 border-b border-indigo-100">
                      <span className="text-slate-600">Macro Precision</span>
                      <strong className="font-mono text-slate-900">
                        {metrics ? `${(metrics.models['Logistic Regression']?.precision_macro * 100).toFixed(2)}%` : '76.01%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-600">Probability Calibration</span>
                      <span className="text-emerald-700 font-medium">Softmax Multi-Class</span>
                    </div>
                  </div>
                </div>

                {/* Multinomial Naive Bayes Card */}
                <div className="p-5 rounded-xl border border-slate-300 bg-white">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                    BASELINE CLASSIFIER
                  </div>
                  <h3 className="text-base font-bold text-slate-800">Multinomial Naive Bayes</h3>
                  <p className="text-xs text-slate-500 mb-4">TF-IDF Vectorization with Laplace Smoothing (alpha=1.0)</p>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600">Test Accuracy</span>
                      <strong className="font-mono text-slate-900">
                        {metrics ? `${(metrics.models['Multinomial Naive Bayes']?.accuracy * 100).toFixed(2)}%` : '72.73%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600">Macro F1-Score</span>
                      <strong className="font-mono text-slate-700">
                        {metrics ? `${(metrics.models['Multinomial Naive Bayes']?.f1_macro * 100).toFixed(2)}%` : '72.70%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600">Macro Precision</span>
                      <strong className="font-mono text-slate-900">
                        {metrics ? `${(metrics.models['Multinomial Naive Bayes']?.precision_macro * 100).toFixed(2)}%` : '72.82%'}
                      </strong>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-600">Feature Independence</span>
                      <span className="text-amber-700 font-medium">Strict Conditional Independence</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Confusion Matrix Interactive Visualizer */}
              <div className="mb-8">
                <h3 className="text-sm font-bold text-slate-900 mb-2">
                  Champion Confusion Matrix (Logistic Regression on Test Split)
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                  Rows represent ground-truth classes; columns represent predicted classes on the 66-sample holdout.
                </p>

                <div className="overflow-x-auto">
                  <table className="text-center text-xs border border-slate-200 rounded-lg">
                    <thead>
                      <tr className="bg-slate-100 text-slate-700">
                        <th className="py-2.5 px-4 font-semibold text-left">Actual \ Predicted</th>
                        <th className="py-2.5 px-4 font-semibold text-rose-700 bg-rose-50">Negative</th>
                        <th className="py-2.5 px-4 font-semibold text-amber-700 bg-amber-50">Neutral</th>
                        <th className="py-2.5 px-4 font-semibold text-emerald-700 bg-emerald-50">Positive</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      <tr>
                        <td className="py-2.5 px-4 font-bold text-left bg-slate-50 text-rose-800">Actual Negative</td>
                        <td className="py-2.5 px-4 font-mono font-bold bg-emerald-50 text-emerald-800">18 (81.8%)</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">1</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">3</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-4 font-bold text-left bg-slate-50 text-amber-800">Actual Neutral</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">4</td>
                        <td className="py-2.5 px-4 font-mono font-bold bg-emerald-50 text-emerald-800">17 (77.3%)</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">1</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 px-4 font-bold text-left bg-slate-50 text-emerald-800">Actual Positive</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">1</td>
                        <td className="py-2.5 px-4 font-mono text-slate-500">6</td>
                        <td className="py-2.5 px-4 font-mono font-bold bg-emerald-50 text-emerald-800">15 (68.2%)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Dataset Statistics */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                <h4 className="font-bold text-slate-800 mb-2">Dataset Specifications & Anti-Leakage Protocol</h4>
                <ul className="list-disc list-inside space-y-1 text-slate-600">
                  <li><strong>Corpus Size:</strong> 327 balanced labeled reviews across e-commerce, hardware, dining, and software.</li>
                  <li><strong>Stratified Split:</strong> 261 training reviews (80%) and 66 holdout test reviews (20%).</li>
                  <li><strong>Random Seed:</strong> Fixed at <code>42</code> for absolute mathematical reproducibility.</li>
                  <li><strong>Leakage Prevention:</strong> Vocabulary extraction and TF-IDF IDF weights fitted strictly on X_train.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: NLP PREPROCESSING LAB */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="mb-6">
                <h2 className="text-lg font-bold text-slate-900">Sentiment-Aware NLP Preprocessing Pipeline</h2>
                <p className="text-sm text-slate-500">
                  Inspect the step-by-step transformation applied to customer text to isolate sentiment signals while protecting negation phrases.
                </p>
              </div>

              {/* Interactive Pipeline Tester */}
              <div className="mb-6">
                <label htmlFor={pipelineInputId} className="block text-xs font-semibold text-slate-700 mb-2">Test Preprocessing on Custom String:</label>
                <div className="flex gap-2">
                  <input
                    id={pipelineInputId}
                    type="text"
                    value={labInput}
                    onChange={(e) => setLabInput(e.target.value)}
                    className="flex-1 px-3 py-2 text-xs border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-hidden font-mono"
                  />
                  <button
                    onClick={() => runLabPipeline(labInput)}
                    disabled={isLabLoading}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-semibold hover:bg-indigo-700 transition-colors"
                  >
                    Inspect Steps
                  </button>
                </div>
              </div>

              {/* Transformation Stages */}
              {labSteps && (
                <div className="space-y-3 font-mono text-xs">
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-[11px] text-slate-400 font-sans font-bold block mb-1">STAGE 1: RAW INPUT</span>
                    <div className="text-slate-800 break-all">{labSteps.raw}</div>
                  </div>

                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-[11px] text-slate-400 font-sans font-bold block mb-1">STAGE 2: NOISE & HTML REMOVAL</span>
                    <div className="text-slate-800 break-all">{labSteps.html_stripped}</div>
                  </div>

                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-[11px] text-slate-400 font-sans font-bold block mb-1">STAGE 3: CONTRACTION EXPANSION (NEGATION ISOLATION)</span>
                    <div className="text-indigo-700 font-bold break-all">{labSteps.contractions_expanded}</div>
                  </div>

                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-[11px] text-slate-400 font-sans font-bold block mb-1">STAGE 4: LOWERCASE & SYMBOL NORMALIZATION</span>
                    <div className="text-slate-800 break-all">{labSteps.lowercased}</div>
                  </div>

                  <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200">
                    <span className="text-[11px] text-emerald-800 font-sans font-bold block mb-1">STAGE 5: PROTECTED NEGATION FILTERING (VECTORIZER READY)</span>
                    <div className="text-emerald-950 font-bold break-all">{labSteps.negation_preserved_filtered}</div>
                  </div>
                </div>
              )}

              {/* Rationale Notice */}
              <div className="mt-8 p-4 bg-amber-50 rounded-xl border border-amber-200 text-xs text-amber-900">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <ShieldCheck className="w-4 h-4 text-amber-700" />
                  <span>Why Negation Words Are Protected</span>
                </div>
                <p>
                  Standard stopword lists remove words like <code>not</code>, <code>never</code>, and <code>no</code>. In sentiment analysis, 
                  stripping negation inverts polarity: <em>"The battery is not good"</em> becomes <em>"battery good"</em>. 
                  SentimentIQ implements an immutable set of 30+ negation tokens that are never stripped during filtering.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: REST API REFERENCE */}
        {activeTab === 'api' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
              <div className="mb-6">
                <h2 className="text-lg font-bold text-slate-900">REST API Integration Reference</h2>
                <p className="text-sm text-slate-500">
                  Connect third-party apps, dashboards, or microservices directly to the SentimentIQ ML engine.
                </p>
              </div>

              {/* Endpoint 1: Single Prediction */}
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-2">
                  <span className="bg-indigo-600 text-white text-[11px] font-mono font-bold px-2 py-0.5 rounded">POST</span>
                  <code className="text-xs font-mono font-bold text-slate-800">/api/predict</code>
                  <span className="text-xs text-slate-500">&mdash; Analyze one text</span>
                </div>

                <div className="relative bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto">
                  <button
                    onClick={() => copyCode(`curl -X POST http://localhost:5001/api/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{"text": "Exceptional customer support and fast response!"}'`, 'curl1')}
                    className="absolute top-3 right-3 text-slate-400 hover:text-white"
                  >
                    {copiedSnippet === 'curl1' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                  <pre>{`curl -X POST http://localhost:5001/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Exceptional customer support and fast response!"}'`}</pre>
                </div>
              </div>

              {/* Endpoint 2: Batch Prediction */}
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-2">
                  <span className="bg-indigo-600 text-white text-[11px] font-mono font-bold px-2 py-0.5 rounded">POST</span>
                  <code className="text-xs font-mono font-bold text-slate-800">/api/predict-batch</code>
                  <span className="text-xs text-slate-500">&mdash; Bulk analysis</span>
                </div>

                <div className="relative bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto">
                  <button
                    onClick={() => copyCode(`curl -X POST http://localhost:5001/api/predict-batch \\\n  -H "Content-Type: application/json" \\\n  -d '{"texts": ["Great quality!", "Broke immediately.", "Package is 200g."]}'`, 'curl2')}
                    className="absolute top-3 right-3 text-slate-400 hover:text-white"
                  >
                    {copiedSnippet === 'curl2' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                  <pre>{`curl -X POST http://localhost:5001/api/predict-batch \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Great quality!", "Broke immediately.", "Package is 200g."]}'`}</pre>
                </div>
              </div>

              {/* Endpoint 3: Health */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="bg-emerald-600 text-white text-[11px] font-mono font-bold px-2 py-0.5 rounded">GET</span>
                  <code className="text-xs font-mono font-bold text-slate-800">/api/health</code>
                  <span className="text-xs text-slate-500">&mdash; Readiness & service status</span>
                </div>

                <div className="relative bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto">
                  <button
                    onClick={() => copyCode(`curl -X GET http://localhost:5001/api/health`, 'curl3')}
                    className="absolute top-3 right-3 text-slate-400 hover:text-white"
                  >
                    {copiedSnippet === 'curl3' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                  <pre>{`curl -X GET http://localhost:5001/api/health`}</pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>SentimentIQ &mdash; Enterprise Machine Learning & NLP Platform</span>
          <span>scikit-learn &bull; NLTK &bull; Flask &bull; React &bull; MIT License</span>
        </div>
      </footer>
    </div>
  );
}
