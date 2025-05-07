# ✨ Emotion Detector

> A fine-tuned Transformer model for detecting emotions from text.  
> Built with Hugging Face Transformers, visualized with Gradio, and deployed on HF Spaces.

---

## 📚 Project Overview

- **Model**: `distilbert-base-uncased` fine-tuned for emotion classification
- **Dataset**: Synthetic dataset generated via LLM (~550 samples)
- **Tech Stack**: Python · Hugging Face · Gradio · Google Colab
- **Deployment**: Model hosted on Hugging Face Hub · Gradio app on Spaces

---

## 🗂️ Project Structure

```Emotion-Detector/
│
├── notebooks/
│ ├── dataset_creation.ipynb # Synthetic data generation
│ ├── model_training.ipynb # Fine-tuning and evaluation
│ └── analyze_model.ipynb
│
├── src/
| ├── data/
│   ├── create_dataset.py # Script for dataset creation
│ ├── models/ 
│   ├── train_model.py # Script for model training
│
├── data/
│ ├── raw.json # Final dataset
│
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore # Files to ignore
```

---

## 📝 Dataset Details

- **Source**: Synthetic (generated with LLM prompting)
- **Format**: JSON (list of dicts with `text` and `emotion` fields)
- **Classes**: Joy, Sadness, Anger, Fear, Surprise
- **Hosted at**: [Hugging Face Datasets Hub](https://huggingface.co/datasets/AliAfkhamii/hf_emotion_generation_texts)

---

## 🧠 Model Training

- **Base Model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Optimization**: AdamW + Weight Decay
- **Training Details**:
  - 6 epochs (early stopping observed at 4)
  - Best validation accuracy ~94%
- **Model Card**: [View on Hugging Face Hub](https://huggingface.co/AliAfkhamii/hf_emotion_detector_text_classifier-distilbert-base-uncased)

---

## ⚙️ Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/AliAfkhamii/emotion-detector-hf.git
cd emotion-detector-hf
```
2. **Create and activate a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install the requirements**:

```bash
pip install -r requirements.txt
```


## 🧹 Professional Practices Followed

*    Version-controlled cleanly with Git

*    Separation of concerns: notebooks → scripts

*    Sensitive credentials (HF tokens) are never hardcoded

*    Dataset loading separated from notebooks

*    Model checkpoints and artifacts managed professionally

*    Progressive cleaning and transformation from exploration → production

## Future Work

*   Add multilingual emotion detection

*   Advanced synthetic data generation

*   Experiment tracking (MLflow or WandB)

*   Dockerization of full app

*   Full MLOps pipeline integration
