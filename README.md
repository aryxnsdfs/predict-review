# predict-review
# BERT Sentiment Classifier (Positive/Negative)

This project uses BERT (`bert-base-uncased`) to classify reviews as **positive** or **negative** using HuggingFace Transformers and PyTorch.

## 📁 Dataset

The dataset used is `TestReviews.csv`, containing:

- `review`: The text
- `class`: 1 (Positive), 0 (Negative)

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/bert-sentiment.git
cd bert-sentiment

# (Optional) Create and activate virtual environment
conda create -n bertsent python=3.10
conda activate bertsent

# Install dependencies
pip install -r requirements.txt

🚀 Run the Classifier
predict_review.py
After training, it will print predictions like:
Positive
Negative

🧠 Model
Uses BertTokenizer and BertForSequenceClassification

Trained for 3 epochs with HuggingFace Trainer

Supports GPU if available


