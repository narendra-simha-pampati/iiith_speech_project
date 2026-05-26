# Multimodal Emotion Recognition on TESS

This repository implements a premium **Multimodal Emotion Recognition** system that classifies human emotions into 7 categories (`angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant_surprise`, `sad`) using **Speech (audio)**, **Text (transcripts)**, and **Multimodal Fusion**. 

The system is trained and evaluated on the **Toronto Emotional Speech Set (TESS)** dataset.

---

## 📂 Project Structure

```
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py          # Speech-only model training
│   │   └── test.py           # Speech-only model evaluation & feature extraction
│   │
│   ├── text_pipeline/
│   │   ├── train.py          # Text-only model training
│   │   └── test.py           # Text-only model evaluation & feature extraction
│   │
│   └── fusion_pipeline/
│       ├── train.py          # Multimodal Fusion training (end-to-end)
│       └── test.py           # Multimodal Fusion evaluation & cluster extraction
│
├── Results/
│   ├── accuracy_table.md     # Markdown table comparing all three model variants
│   ├── silhouette_analysis.md # Quantitative cluster separability analysis
│   └── error_analysis.md     # 5 representative failure cases from the fusion model
│
├── utils.py                  # Audio/text preprocessing, tokenization, and data splitting
├── models_def.py             # Reusable PyTorch model class definitions
├── cache_data.py             # Feature caching script (loads audio once and saves to disk)
├── run_experiments.py        # Master orchestrator script (runs all training & testing)
├── requirements.txt          # Python dependencies
└── Report.md                 # Comprehensive project report (Architecture, Experiments, Analysis)
```

---

## 🛠️ Architecture Decisions & Design

The system implements the 5 core functional blocks:

1. **Preprocessing**:
   - **Speech**: Audio is resampled to 16,000Hz, silent intervals are trimmed using standard energy thresholding (`librosa.effects.trim` at 20dB), and signals are padded/truncated to a uniform duration of 2.0 seconds (32,000 samples).
   - **Text**: The target phrase is set as `"say the word {word}"`. Tokens are lowercased and mapped to vocabulary indices.
2. **Feature Extraction**:
   - **Speech**: Extracts **40 Mel-Frequency Cepstral Coefficients (MFCCs)** over time frames, capturing the short-term power spectrum of the speech signal.
   - **Text**: A learned dense word embedding layer (`nn.Embedding`, dimension 64) mapping each token to a vector space.
3. **Temporal/Contextual Modelling**:
   - **Speech**: A **Bidirectional GRU** (2 layers, 64 hidden units) models temporal dynamics of voice prosody, pitch, and tone over time steps.
   - **Text**: A **Bidirectional GRU** (2 layers, 64 hidden units) models the semantic and contextual emotional meaning across tokens.
4. **Fusion**:
   - **Early Fusion (Feature Concatenation)**: The speech temporal vector (128-dim) and text contextual vector (128-dim) are concatenated to form a unified multimodal representation (256-dim), which is passed through a non-linear Fusion MLP (`nn.Linear` $\to$ `ReLU` $\to$ `Dropout`) to output a fused 128-dimensional embedding.
5. **Classifier**:
   - Fully Connected Layers mapping the learned representations to log-probabilities over the 7 emotion categories.

---

## ⚡ Setup & Run Instructions

### 1. Environment Setup
Make sure you have Python 3.9+ and activate your virtual environment:
```bash
# Activate your python virtual environment
source /Applications/Projects/IIITH_project/Indic-Accent-Identification/venv/bin/activate

# Install dependencies
pip install -r project/requirements.txt
```

### 2. Feature Caching (Critical Optimization)
To avoid loading and extracting MFCCs for 2,800 audio files repeatedly (which is heavy and slow), run the caching script **once**. It automatically performs deduplication, sets up the disjoint splits, extracts features, and saves them to a highly optimized serialized file:
```bash
python project/cache_data.py
```
*Note: This script has already been run successfully and the cached file `project/cached_data.pkl` is fully generated!*

### 3. Run All Experiments & Analyses
To train all three models (Speech, Text, Fusion) sequentially, run testing, compute the quantitative separability metrics (Silhouette Scores), and perform error analysis:
```bash
python project/run_experiments.py
```

This single command will:
1. Train and test the Speech-only pipeline (saving weights and test metrics).
2. Train and test the Text-only pipeline.
3. Train and test the Multimodal Fusion pipeline.
4. Populate the `project/Results/` folder with complete metrics, silhouette scores, and failure analyses.

---

## 📊 Summary of Scientific Insights

1. **Speech Modality is King**: TESS is a high-fidelity acted speech database where actors express emotions purely through vocal dynamics (prosody, pitch variations, speed, and volume). The Speech model easily separates emotions, reaching **~98% test accuracy**.
2. **Text Modality is Semantically Neutral**: The text transcripts (e.g., `"say the word back"`, `"say the word date"`) contain zero semantic emotional information. Since we enforce a strict **word-disjoint split** (unseen words in the test set), the Text-only model performs exactly like a **random guess (~14.3% accuracy)**.
3. **Multimodal Robustness**: In the Fusion model, the network learns to weight the speech modality heavily and ignore the neutral text features, preserving the outstanding classification performance (**~98% accuracy**).

For a deep dive into the experimental logs, cluster separability metrics, and error analysis, please check [Report.md](file:///Applications/Projects/iiith_final/project/Report.md).
