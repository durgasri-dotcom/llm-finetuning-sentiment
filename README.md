# LoRA Fine-Tuning: Sentiment Classification with PEFT

Parameter-efficient fine-tuning of DistilBERT for sentiment classification using Low-Rank Adaptation (LoRA) via HuggingFace PEFT library.

## Results

| Metric   | Epoch 1 | Epoch 2 | Epoch 3    |
| -------- | ------- | ------- | ---------- |
| Accuracy | 87.4%   | 86.6%   | Best model |
| F1 Score | 0.874   | 0.865   | 0.874      |
| Loss     | 0.6448  | 0.3015  | 0.2555     |

**Trainable parameters:** 887,042 / 67,842,052 (1.31%) — LoRA efficiency demonstrated

## Architecture

- **Base model:** `distilbert-base-uncased`
- **Dataset:** IMDB Sentiment (2,000 train / 500 eval)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
- **LoRA config:** r=16, alpha=32, dropout=0.1
- **Target modules:** `q_lin`, `v_lin` (attention layers)
- **Task:** Sequence Classification (Binary: Positive/Negative)

## Key Design Decisions

**Why LoRA?**
LoRA freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into the attention layers. This reduces trainable parameters by ~98% while achieving competitive performance critical for resource-constrained environments and rapid experimentation.

**Why DistilBERT?**
DistilBERT retains 97% of BERT's performance at 40% smaller size, making it ideal for demonstrating fine-tuning methodology without excessive compute requirements.

**Evaluation methodology:**

- Cross-epoch evaluation after every epoch
- Metrics: Accuracy + Weighted F1 (handles class imbalance)
- Best model checkpoint saved based on F1 score
- Loss curve monitored for overfitting detection

## Tech Stack

```
Python 3.11
HuggingFace Transformers
PEFT (Parameter Efficient Fine-Tuning)
PyTorch
datasets
scikit-learn
accelerate
```

## Setup & Run

```bash
# Clone repo
git clone https://github.com/durgasri-dotcom/llm-finetuning-sentiment.git
cd llm-finetuning-sentiment

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install transformers datasets peft torch accelerate scikit-learn huggingface_hub

# Run training
python train.py
```

## Experimental Findings

- LoRA converges faster than full fine-tuning on small datasets
- Loss dropped from 0.6448 (epoch 0.4) to 0.2555 (epoch 2.8) — 60% reduction
- Minimal overfitting observed between train and eval loss curves
- Parameter efficiency: only 1.31% of weights trained, achieving 87.4% accuracy

## Relevance to LLM Research

This project demonstrates:

- **Parameter-efficient fine-tuning** — core technique in modern LLM post-training
- **Experimental design** — hypothesis, dataset selection, evaluation methodology, result analysis
- **Hallucination-adjacent research** — understanding how fine-tuning affects model confidence and output reliability
- **Scalable training patterns** — applicable to larger models (LLaMA, Mistral) with same PEFT approach

## Project Structure

llm-finetuning-sentiment/
├── train.py # LoRA fine-tuning script
├── evaluate.py # Inference + evaluation on new samples
├── requirements.txt # Pinned dependencies for reproducibility
├── .gitignore # Excludes venv, model weights, cache
└── README.md # Project documentation

## Run Inference

After training, evaluate the model on sample texts:

```bash
python evaluate.py
```

Runs prediction on 8 diverse test samples and enters interactive mode where you can type any text and get sentiment prediction with confidence scores.

## Author

Sri Durga Abhigna Tanguturi

- GitHub: [durgasri-dotcom](https://github.com/durgasri-dotcom)
- LinkedIn: [durgasritanguturi](https://linkedin.com/in/durgasritanguturi)
- CyberMind AI Platform: [cybermindai.streamlit.app](https://cybermindai.streamlit.app)
