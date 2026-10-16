// SentimentIQ Frontend Client Logic

let sessionCounts = { Positive: 0, Negative: 0, Neutral: 0, Total: 0 };
let currentBatchData = [];

// Sample texts
const samples = [
  "The sound quality is crystal clear with deep bass and excellent battery life. Absolutely love this headset!",
  "Worst customer service experience of my life. The item arrived broken and nobody responded to my emails.",
  "The package arrived on Tuesday as scheduled. The box was sealed and contents were inside."
];

function loadSample(index) {
  const textarea = document.getElementById("textInput");
  textarea.value = samples[index] || "";
  updateCharCount();
  analyzeSentiment();
}

// Real-time debounced typing predictor
let typingTimer = null;
const DEBOUNCE_DELAY_MS = 400;

function handleTextInput() {
  updateCharCount();
  const text = document.getElementById("textInput").value.trim();

  if (typingTimer) clearTimeout(typingTimer);

  if (!text) {
    document.getElementById("resultBox").classList.add("hidden");
    return;
  }

  // Set typing indicator
  const btnText = document.getElementById("btnText");
  btnText.innerText = "Typing...";

  typingTimer = setTimeout(() => {
    analyzeSentiment(true);
  }, DEBOUNCE_DELAY_MS);
}

document.getElementById("textInput")?.addEventListener("input", handleTextInput);

function clearInput() {
  document.getElementById("textInput").value = "";
  updateCharCount();
  document.getElementById("resultBox").classList.add("hidden");
}

async function analyzeSentiment(isSilent = false) {
  const text = document.getElementById("textInput").value.trim();
  if (!text) {
    if (!isSilent) alert("Please enter a sentence or review to analyze.");
    return;
  }

  const btnText = document.getElementById("btnText");
  const spinner = document.getElementById("btnSpinner");
  const analyzeBtn = document.getElementById("analyzeBtn");

  try {
    if (!isSilent) {
      analyzeBtn.disabled = true;
      spinner.classList.remove("hidden");
      btnText.innerText = "Analyzing...";
    } else {
      btnText.innerText = "Inferring...";
    }

    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Prediction request failed.");
    }

    displayResult(data);
    updateSessionStats(data.sentiment);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    spinner.classList.add("hidden");
    btnText.innerText = "Analyze Sentiment";
  }
}

function displayResult(res) {
  const resultBox = document.getElementById("resultBox");
  const badge = document.getElementById("sentimentBadge");
  const label = document.getElementById("sentimentLabel");
  const conf = document.getElementById("confidenceVal");
  const explanation = document.getElementById("explanationText");
  const tokenBox = document.getElementById("tokenPills");

  badge.className = `sentiment-pill-large ${res.sentiment}`;
  label.innerText = res.sentiment;
  conf.innerText = `${(res.confidence * 100).toFixed(1)}%`;
  explanation.innerText = res.explanation;

  // Probabilities
  const probs = res.probabilities;
  const pPos = ((probs.Positive || 0) * 100).toFixed(1);
  const pNeg = ((probs.Negative || 0) * 100).toFixed(1);
  const pNeu = ((probs.Neutral || 0) * 100).toFixed(1);

  document.getElementById("barPos").style.width = `${pPos}%`;
  document.getElementById("valPos").innerText = `${pPos}%`;

  document.getElementById("barNeg").style.width = `${pNeg}%`;
  document.getElementById("valNeg").innerText = `${pNeg}%`;

  document.getElementById("barNeu").style.width = `${pNeu}%`;
  document.getElementById("valNeu").innerText = `${pNeu}%`;

  // Tokens
  tokenBox.innerHTML = "";
  if (res.influential_tokens && res.influential_tokens.length > 0) {
    res.influential_tokens.forEach(t => {
      const span = document.createElement("span");
      span.className = "token-tag";
      span.innerText = `${t.token} (${t.weight > 0 ? "+" : ""}${t.weight})`;
      tokenBox.appendChild(span);
    });
  }

  resultBox.classList.remove("hidden");
}

function updateSessionStats(sentiment) {
  sessionCounts.Total++;
  if (sessionCounts[sentiment] !== undefined) {
    sessionCounts[sentiment]++;
  }

  document.getElementById("totalAnalyzedCount").innerText = sessionCounts.Total;
  document.getElementById("totalPosCount").innerText = sessionCounts.Positive;
  document.getElementById("totalNegCount").innerText = sessionCounts.Negative;
  document.getElementById("totalNeuCount").innerText = sessionCounts.Neutral;

  const pPos = ((sessionCounts.Positive / sessionCounts.Total) * 100).toFixed(1);
  const pNeg = ((sessionCounts.Negative / sessionCounts.Total) * 100).toFixed(1);
  const pNeu = ((sessionCounts.Neutral / sessionCounts.Total) * 100).toFixed(1);

  document.getElementById("posPercentage").innerText = `${pPos}% of session`;
  document.getElementById("negPercentage").innerText = `${pNeg}% of session`;
  document.getElementById("neuPercentage").innerText = `${pNeu}% of session`;
}

// CSV Batch Handling
function handleFileSelect(evt) {
  const file = evt.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    parseCSV(text);
  };
  reader.readAsText(file);
}

let parsedCSVRows = [];
let csvHeaders = [];

function parseCSV(csvText) {
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) {
    alert("CSV must contain at least a header and one data row.");
    return;
  }

  // Parse header
  csvHeaders = lines[0].split(",").map(h => h.trim().replace(/^["']|["']$/g, ""));
  parsedCSVRows = [];

  for (let i = 1; i < lines.length; i++) {
    // Simple CSV row parser (handles quotes)
    const regex = /(?:,|\n|^)("(?:(?:"")*[^"]*)*"|[^",\n]*|(?:\n|$))/g;
    const row = [];
    let match;
    while ((match = regex.exec(lines[i])) !== null && match[0] !== "") {
      let val = match[1] || "";
      if (val.startsWith('"') && val.endsWith('"')) {
        val = val.substring(1, val.length - 1).replace(/""/g, '"');
      }
      row.push(val);
      if (regex.lastIndex >= lines[i].length) break;
    }
    if (row.length > 0) {
      parsedCSVRows.push(row);
    }
  }

  const select = document.getElementById("csvColumnSelect");
  select.innerHTML = "";
  csvHeaders.forEach((h, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.innerText = `${h} (Column ${idx + 1})`;
    // Auto-select text/review columns
    if (/text|review|comment|feedback|content/i.test(h)) {
      opt.selected = true;
    }
    select.appendChild(opt);
  });

  document.getElementById("csvOptionsBox").classList.remove("hidden");
}

async function runBatchAnalysis() {
  const colIdx = parseInt(document.getElementById("csvColumnSelect").value, 10);
  const texts = parsedCSVRows.map(r => r[colIdx] || "").filter(t => t.trim().length > 0);

  if (texts.length === 0) {
    alert("Selected column has no valid text entries.");
    return;
  }

  try {
    const res = await fetch("/api/predict-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Batch prediction failed.");
    }

    currentBatchData = data.items;
    renderBatchTable(data.items);
    document.getElementById("batchResultsBox").classList.remove("hidden");
  } catch (err) {
    alert("Batch error: " + err.message);
  }
}

function renderBatchTable(items) {
  const tbody = document.getElementById("batchTableBody");
  tbody.innerHTML = "";

  items.slice(0, 50).forEach((item, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td style="max-width: 400px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.text}</td>
      <td><span class="pill-btn ${item.sentiment.toLowerCase()}">${item.sentiment}</span></td>
      <td>${(item.confidence * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

function exportBatchCSV() {
  if (!currentBatchData || currentBatchData.length === 0) {
    alert("No batch data to export.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,Original Text,Predicted Sentiment,Confidence,Cleaned Text\n";
  currentBatchData.forEach(item => {
    const safeText = `"${(item.text || '').replace(/"/g, '""')}"`;
    const safeCleaned = `"${(item.cleaned_text || '').replace(/"/g, '""')}"`;
    csvContent += `${safeText},${item.sentiment},${(item.confidence * 100).toFixed(1)}%,${safeCleaned}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "sentimentiq_predictions.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
