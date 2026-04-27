# 🩺 MedBot — AI-Powered Medical Symptom Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)

**A full-stack medical chatbot that combines ML-based disease prediction with a medical Q&A retrieval engine — all in a sleek, glassmorphism UI.**

</div>

---

## ✨ Features

- **🔬 Disease Prediction** — Trained on 246,000+ samples across 773 diseases and 377 symptoms using a soft-voting ensemble of two `ExtraTreesClassifier` models.
- **📚 Medical Q&A** — Hybrid TF-IDF retrieval (word + char n-grams) over the MedQuAD dataset to answer medical knowledge questions.
- **🧠 Intent Detection** — Automatically routes messages between greeting, symptom report, medical question, and non-medical categories.
- **💊 Symptom Extraction** — NLP pipeline using alias maps, regex keyword patterns, body-area pain detection, and the full CSV-derived symptom vocabulary.
- **🚨 Urgent Symptom Alerts** — Immediate escalation prompts for critical symptoms (e.g. chest pain, difficulty breathing).
- **📊 Confidence Meter** — Visual ML confidence bar displayed with every disease prediction.
- **🎨 Glassmorphism UI** — Animated dark-mode chat interface with sidebar, particle canvas background, prompt chips, and smooth card animations.
- **🔁 Health Endpoint** — `/health` JSON endpoint to check model status, vocabulary size, and Q&A entry count.

---

## 🗂️ Project Structure

```
medical_chatbot/
│
├── app.py                        # Flask app — routing, intent dispatch, API endpoints
├── train_model.py                # Model training — disease ensemble + Q&A vectorizers
├── requirements.txt              # Python dependencies
│
├── data/                         # ⚠️  NOT included — download manually (see below)
│   ├── Final_Augmented_dataset_Diseases_and_Symptoms.csv
│   └── medquad.csv
│
├── model/                        # Auto-created on training
│   ├── disease_model.pkl
│   ├── label_encoder.pkl
│   ├── symptom_features.pkl
│   ├── disease_centroids.pkl
│   ├── disease_precautions.pkl
│   ├── qa_word_vectorizer.pkl
│   ├── qa_char_vectorizer.pkl
│   └── qa_entries.pkl
│
├── utils/
│   ├── __init__.py
│   ├── intent_detector.py        # Classifies user message intent
│   ├── symptom_extractor.py      # NLP symptom extraction pipeline
│   ├── model_predictor.py        # Loads ensemble + runs disease inference
│   ├── qa_matcher.py             # Loads TF-IDF + performs Q&A retrieval
│   ├── response_generator.py     # Builds structured JSON response payloads
│   ├── medical_knowledge.py      # Symptom aliases, normalization, default prompts
│   └── light_ensemble.py         # Lightweight soft-voting ensemble wrapper
│
├── templates/
│   └── index.html                # Jinja2 HTML template (full chat UI)
│
└── static/
    ├── script.js                 # Frontend chat logic, rendering, animations
    └── style.css                 # Glassmorphism dark-mode stylesheet
```

---

## 📦 Datasets — Download Required

> **The `data/` folder is not included in this repository due to file size.** You must download the datasets manually and place them inside a `data/` folder in the project root.

### 1. Disease–Symptom Dataset (Large — 190 MB)

| Detail | Info |
|---|---|
| **Source** | Kaggle — [Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) |
| **File name** | `Final_Augmented_dataset_Diseases_and_Symptoms.csv` |
| **Rows** | ~246,000 (augmented) |
| **Diseases** | 773 |
| **Symptoms** | 377 (one-hot encoded columns) |
| **Size** | ~190.79 MB |

### 2. MedQuAD — Medical Q&A Dataset

| Detail | Info |
|---|---|
| **Source** | [MedQuAD on GitHub](https://github.com/abachaa/MedQuAD) or [Kaggle mirror](https://www.kaggle.com/datasets/pythonapipackages/medquad) |
| **File name** | `medquad.csv` |
| **Columns** | `question`, `answer`, `focus_area` |
| **Purpose** | Answers factual medical knowledge questions |

### 📁 Place the files as follows:

```
medical_chatbot/
└── data/
    ├── Final_Augmented_dataset_Diseases_and_Symptoms.csv   ← from Kaggle
    └── medquad.csv                                          ← from MedQuAD
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- At least **8 GB RAM** (required for training; inference is much lighter)
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical_chatbot.git
cd medical_chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Datasets

Download both datasets (see above) and place them in the `data/` folder.

### 5. Train the Models

```bash
# Train both disease prediction + Q&A models
python train_model.py

# Or train individually
python train_model.py disease
python train_model.py qa
```

> ⏱️ **Training time** on an 8 GB laptop: ~25–35 minutes total.
> Expected accuracy:
> - Disease model Top-1: ~78–85% | Top-3: ~91–95%
> - Q&A model topic match (Top-3): high recall

### 6. Run the App

```bash
python app.py
```

Open your browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🚀 Usage

Once the app is running, you can:

- **Describe symptoms** — e.g. `"I have fever, headache and cough for 2 days"`
- **Ask medical questions** — e.g. `"What causes Urinary Tract Infections?"`
- **Use prompt chips** in the sidebar for quick queries
- **Check model status** at [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

### Example Chat Interactions

```
User: I have chest pain and difficulty breathing
Bot:  🚨 Urgent — These symptoms may require immediate attention...

User: What causes diabetes?
Bot:  📚 Medical Information — [MedQuAD answer]

User: I have fever, joint pain, and rash
Bot:  🔬 Symptom Analysis — Top prediction: [Disease] (confidence: 82%)
```

---

## 🧠 How It Works

```
User Message
     │
     ▼
Intent Detector ──────────────────────────────────────────────────┐
     │                                                             │
     ├── greeting     → Greeting response                         │
     ├── non_medical  → Redirect response                         │
     ├── question     → QA Matcher → (fallback) Disease Predictor │
     └── symptom      → Symptom Extractor → Disease Predictor     │
                                                                   │
                              Structured JSON Response ◄───────────┘
                                        │
                              Flask → Frontend (script.js)
                                        │
                              Rendered Card in Chat UI
```

### ML Pipeline

1. **Symptom Extraction** — The NLP pipeline maps natural language text to the 377 one-hot symptom columns using aliases, regex patterns, body-area pain maps, and misspelling rescue.
2. **Disease Prediction** — A soft-voting ensemble of two `ExtraTreesClassifier` models (seed=42 and seed=99, depth=25, 60 trees each) predicts the top disease with probability scores.
3. **Centroid Matching** — Cosine similarity to precomputed disease centroids provides a second confidence signal.
4. **Q&A Retrieval** — Hybrid word (1–3 gram) + character (3–5 gram) TF-IDF vectorizers find the closest MedQuAD entry using sparse matrix cosine similarity.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML / NLP | scikit-learn, NumPy, Pandas, SciPy |
| Disease Model | `ExtraTreesClassifier` (soft-voting ensemble) |
| Q&A Model | TF-IDF (word + char n-grams), cosine similarity |
| Frontend | Vanilla JS, HTML5 Canvas (particles), CSS3 |
| Styling | Glassmorphism dark theme, CSS animations |
| Data | Kaggle Diseases-Symptoms, MedQuAD |

---

## ⚠️ Disclaimer

> **MedBot is a student mini-project for educational purposes only.**
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified doctor or healthcare provider for any medical concerns.

---

## 👥 Team

This project was built as a group mini-project by 5 members:

| Member | Role |
|---|---|
| **Member 1** | *(Pratik Bhattacharjee)* |
| **Member 2** | *(Ebrahim Balakhia)* |
| **Member 3** | *(Shivam Burkul)* |
| **Member 4** | *(Faiz Chauhan)* |
| **Member 5** | *(Rajeevkumar Chauhan)* |

> All members contributed equally to the development of this project.

---

## 📄 License

This project is for academic/educational use. Dataset licenses are governed by their respective sources:
- [Kaggle Diseases-Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)
- [MedQuAD](https://github.com/abachaa/MedQuAD) — see original repository for license details.