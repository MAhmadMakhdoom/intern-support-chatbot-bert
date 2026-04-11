# Intern Support Chatbot — BERT + HuggingFace NLP

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple?style=flat-square)
![BERT](https://img.shields.io/badge/Model-BERT--base--uncased-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What This Project Does

This chatbot automates responses to common intern questions using:
- **BERT** fine-tuned for 7-class intent classification
- **133 training samples** — FAQ documents + historical support tickets
- **Real-time answers** via a professional Gradio UI
- **Voice input** powered by OpenAI Whisper
- **96% accuracy** on held-out test set

---

## Intents Covered

| Intent | Example Questions |
|---|---|
| `working_hours` | "What time does office open?" / "Do we work Saturdays?" |
| `leave_request` | "How do I apply for leave?" / "I need a sick day" |
| `stipend_query` | "When do I get my salary?" / "I haven't received stipend" |
| `it_support` | "My laptop is broken" / "Can't connect to wifi" |
| `credential_issue` | "Forgot my password" / "Account is locked" |
| `hr_policy` | "What is dress code?" / "Will I get a certificate?" |
| `general_query` | "Who is my supervisor?" / "Where is HR office?" |

---

## Dataset

Built from scratch — two sources combined:

**Source 1 — FAQ Documents (98 rows)**
- 7 intents × 14 question variations each
- Realistic first-week intern questions

**Source 2 — Historical Support Tickets (35 rows)**
- 7 intents × 5 tickets each (TKT001–TKT035)
- Adds real conversational variety

**Total: 133 samples | Train: 106 | Test: 27 | Balanced: 19 per intent**

---

## Model Performance (133 samples, 15 epochs, GPU T4)

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| credential_issue | 0.75 | 1.00 | 0.86 |
| general_query | 1.00 | 1.00 | 1.00 |
| hr_policy | 1.00 | 0.75 | 0.86 |
| it_support | 1.00 | 1.00 | 1.00 |
| leave_request | 1.00 | 1.00 | 1.00 |
| stipend_query | 1.00 | 1.00 | 1.00 |
| working_hours | 1.00 | 1.00 | 1.00 |
| **Overall Accuracy** | | | **96%** |

Training loss: 1.88 → 0.15 over 15 epochs

---

## System Architecture

```
User Input (text or voice)
       │
       ▼
Whisper STT (voice only)
       │
       ▼
Text Preprocessing (lowercase, clean)
       │
       ▼
BERT Tokenizer (bert-base-uncased, max_len=64)
       │
       ▼
Fine-tuned BERT Classifier (7 classes)
       │
       ▼
Softmax → Confidence Score
       │
  ┌────┴────┐
≥0.45    <0.45
  │          │
Answer    Fallback
Lookup    Message
  │
  ▼
Gradio Chat UI
```

---

## Project Structure

```
intern-support-chatbot-bert/
├── src/
│   ├── data_builder.py     # Dataset creation + preprocessing
│   ├── train.py            # BERT fine-tuning pipeline
│   ├── predict.py          # Inference + confidence scoring
│   └── app.py              # Full Gradio UI with voice
├── data/
│   ├── intern_dataset.csv          # FAQ base (98 rows)
│   ├── support_tickets.csv         # Ticket data (35 rows)
│   └── intern_dataset_full.csv     # Combined (133 rows)
├── model/
│   └── README.md           # How to get model weights
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start (Google Colab)

```python
# 1. Open Colab → Upload intern_chatbot.ipynb
# 2. Runtime → Change runtime type → T4 GPU
# 3. Runtime → Run all (Ctrl+F9)
# 4. Copy the Gradio public URL printed at the bottom
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | BERT (bert-base-uncased) |
| Training | HuggingFace Trainer API |
| Deep Learning | PyTorch |
| Voice | OpenAI Whisper (tiny) |
| UI | Gradio Blocks |
| Data | Pandas + Scikit-learn |
| Platform | Google Colab (T4 GPU) |
| Storage | Google Drive |

---

## Author

**Muhammad Ahmad Makhdoom** — AI Engineering Student

Built as an internship support automation project demonstrating a complete
end-to-end NLP pipeline: data collection → BERT fine-tuning → deployed UI.

---

## License

MIT License — free to use with attribution.
