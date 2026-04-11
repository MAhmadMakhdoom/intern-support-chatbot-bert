"""
predict.py
Inference module — loads fine-tuned BERT and predicts intent + answer.

Usage:
    from predict import load_model, predict_intent
    model, tokenizer, device = load_model("../model")
    intent, confidence, answer = predict_intent("When do I get paid?", ...)
"""
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

CONFIDENCE_THRESHOLD = 0.45

ANSWERS = {
    "working_hours": (
        "Core working hours are 9am to 6pm, Monday to Friday. "
        "Saturday and Sunday are off. Lunch break is from 1pm to 2pm. "
        "Flexible timing can be discussed with your supervisor after the first month."
    ),
    "leave_request": (
        "Submit a leave request through the HR portal at least 3 days in advance. "
        "Your supervisor will approve it via email. "
        "For sick leave, inform your supervisor by 9am on the same day."
    ),
    "stipend_query": (
        "Intern stipends are processed on the 25th of every month "
        "and credited within 2 working days. "
        "Ensure your bank details are submitted to HR in your first week."
    ),
    "it_support": (
        "Visit the IT helpdesk on Floor 2 with your employee ID. "
        "For urgent issues call ext 200 or email it-support@company.com."
    ),
    "credential_issue": (
        "Your login credentials are sent to your personal email before joining. "
        "If not received, contact IT at it-support@company.com. "
        "Password resets can be done via the portal login page."
    ),
    "hr_policy": (
        "Interns follow business casual dress code. "
        "You must sign an NDA on your first day. "
        "Internship certificates are issued within 2 weeks of completion."
    ),
    "general_query": (
        "Your assigned buddy is your first point of contact. "
        "HR is reachable at hr@company.com or Floor 3. "
        "Your access card is issued on day one at reception."
    ),
}

FALLBACK = (
    "I am not confident about that question. You can ask me about:\n"
    "- Working hours\n- Leave requests\n- Stipend / salary\n"
    "- IT support\n- Login credentials\n- HR policies\n- General office queries"
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_path):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def predict_intent(text, model, tokenizer, le, device,
                   answers=ANSWERS, threshold=CONFIDENCE_THRESHOLD):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, max_length=64, padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    pred_idx   = torch.argmax(probs, dim=1).item()
    intent     = le.inverse_transform([pred_idx])[0]
    answer     = answers.get(intent, FALLBACK) if confidence >= threshold else FALLBACK
    return intent, confidence, answer
