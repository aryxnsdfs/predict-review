import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. Load Dataset
df = pd.read_csv(r'c:\Users\aryan\Downloads\TestReviews.csv', encoding='latin1')[['review', 'class']]
df.columns = ['text', 'label']

# 2. Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42
)

# 3. Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encoding = tokenizer(train_texts.tolist(), max_length=200, padding=True, truncation=True)
test_encoding = tokenizer(test_texts.tolist(), max_length=200, padding=True, truncation=True)

# 4. Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

traindata = ReviewDataset(train_encoding, train_labels)
testdata = ReviewDataset(test_encoding, test_labels)

# 5. Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_eval_batch_size=64,
    per_device_train_batch_size=16,
    eval_strategy='epoch',
    logging_dir='./logs',
    save_strategy='epoch',
    logging_steps=10,
    load_best_model_at_end=True,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=traindata,
    eval_dataset=testdata,
)

trainer.train()

# 8. Prediction
def predict_review(text):
    model.eval()  # Put model in evaluation mode

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input and move tensors to the same device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=200)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient computation for faster inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

print(predict_review("Absolutely loved the ambiance and the service was top-notch!"))
print(predict_review("Would not recommend. Completely disappointed with the service."))
