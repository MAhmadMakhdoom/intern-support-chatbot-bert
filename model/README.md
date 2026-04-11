# Model Weights

Model weights are NOT stored here (440 MB exceeds GitHub limit).

## How to get the model

### Option 1 — Train from scratch (~3 min on Colab T4 GPU)
Open `intern_chatbot.ipynb` in Colab, enable T4 GPU, run all cells.
Model saves automatically to Google Drive at:
`/content/drive/MyDrive/intern_chatbot/model/`

### Option 2 — Load from Drive after training
```python
from transformers import BertTokenizer, BertForSequenceClassification
model_path = '/content/drive/MyDrive/intern_chatbot/model'
tokenizer  = BertTokenizer.from_pretrained(model_path)
model      = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
```

## Model Specs

| Property | Value |
|---|---|
| Base model | bert-base-uncased |
| Type | BertForSequenceClassification |
| Labels | 7 |
| Max sequence length | 64 tokens |
| Training samples | 133 |
| Training epochs | 15 |
| Final accuracy | 96% |
| Framework | PyTorch + HuggingFace |
