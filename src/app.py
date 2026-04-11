"""
app.py
Professional Gradio UI for the Intern Support Chatbot.
Loads fine-tuned BERT + Whisper voice input.

Usage:
    python app.py
"""
import re
import torch
import whisper
import gradio as gr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from predict import predict_intent, load_model, ANSWERS, FALLBACK

MODEL_PATH = "../model"
DATA_PATH  = "../data/intern_dataset_full.csv"
THRESHOLD  = 0.45

# ── load models ───────────────────────────────────────────
print("Loading BERT model...")
model, tokenizer, device = load_model(MODEL_PATH)

print("Loading LabelEncoder...")
df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
le.fit(df["intent"])

print("Loading Whisper (tiny)...")
whisper_model = whisper.load_model("tiny")
print("All models ready.")


# ── chat logic ────────────────────────────────────────────
def respond(message, history):
    if not message.strip():
        return history, ""
    intent, confidence, answer = predict_intent(
        message, model, tokenizer, le, device,
        answers=ANSWERS, threshold=THRESHOLD
    )
    response = (
        f"{answer}\n\n"
        f"*Detected intent: **{intent}** · Confidence: {confidence:.0%}*"
    )
    history.append((message, response))
    return history, ""


def handle_voice(audio, history):
    if audio is None:
        history.append(("Voice", "No audio detected. Please record again."))
        return history, ""
    try:
        result = whisper_model.transcribe(audio, language="en")
        text   = result["text"].strip()
        if not text:
            history.append(("Voice", "Could not understand. Please speak clearly."))
            return history, ""
        return respond(text, history)
    except Exception as e:
        history.append(("Voice", f"Voice error: {str(e)}"))
        return history, ""


def clear_chat():
    return [], ""


# ── CSS ───────────────────────────────────────────────────
css = """
.gradio-container {
    max-width: 880px !important; margin: auto !important;
    font-family: 'Segoe UI', sans-serif !important;
    background: #f7f8fc !important;
}
#header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 30px 32px; border-radius: 16px; margin-bottom: 18px; text-align: center;
}
#header h1 { color:#fff !important; font-size:28px !important; font-weight:700 !important; }
#header p  { color:#a0aec0 !important; font-size:13px !important; }
.badge {
    display:inline-block; background:rgba(255,255,255,0.12); color:#e2e8f0;
    border-radius:20px; padding:4px 14px; font-size:12px; margin:3px;
    border:1px solid rgba(255,255,255,0.2);
}
#chatbox { border-radius:16px !important; border:1px solid #e2e8f0 !important;
           box-shadow:0 2px 20px rgba(0,0,0,0.06) !important; background:#fff !important; }
#send-btn  { background:#0f3460 !important; color:white !important;
             border-radius:12px !important; font-weight:600 !important;
             font-size:15px !important; min-height:48px !important; border:none !important; }
#voice-btn { background:#2d3748 !important; color:white !important;
             border-radius:12px !important; font-weight:600 !important; border:none !important; }
#clear-btn { background:#fff !important; color:#e53e3e !important;
             border:1.5px solid #e53e3e !important; border-radius:12px !important;
             font-weight:600 !important; }
#status-bar { text-align:center; color:#a0aec0; font-size:12px; margin-top:10px;
              padding:8px; background:#fff; border-radius:8px; border:1px solid #e2e8f0; }
"""

# ── UI ────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Intern Support Chatbot") as demo:

    gr.HTML("""
    <div id="header">
        <h1>Intern Support Chatbot</h1>
        <p>Powered by BERT · HuggingFace Transformers · Real-time intent detection</p>
        <div>
            <span class="badge">Leave</span>
            <span class="badge">Salary</span>
            <span class="badge">IT Support</span>
            <span class="badge">HR Policy</span>
            <span class="badge">Credentials</span>
            <span class="badge">Working Hours</span>
        </div>
    </div>
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbox", height=440, bubble_full_width=False, show_label=False,
        avatar_images=(
            "https://api.dicebear.com/7.x/thumbs/svg?seed=intern",
            "https://api.dicebear.com/7.x/bottts/svg?seed=bot",
        ),
        value=[(None, "Hello! I am your Intern Support Assistant. How can I help you today?")],
    )

    with gr.Row():
        msg      = gr.Textbox(placeholder="Type your question here...",
                              scale=9, show_label=False, container=False, lines=1)
        send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

    with gr.Row():
        voice_input = gr.Audio(sources=["microphone"], type="filepath",
                               label="Speak your question", scale=8)
        voice_btn   = gr.Button("Submit Voice", elem_id="voice-btn", scale=2)

    gr.Examples(
        examples=[
            "When do I get my stipend?",
            "How do I apply for leave?",
            "My laptop is not working",
            "What are the working hours?",
            "I forgot my password",
            "What is the dress code?",
            "Who is my supervisor?",
            "Will I get a certificate?",
        ],
        inputs=msg, label="",
    )

    clear_btn = gr.Button("Clear Chat", elem_id="clear-btn")

    gr.HTML("""
    <div id="status-bar">
        BERT · 7 intents · 133 training samples · 96% accuracy · Voice enabled (Whisper)
    </div>
    """)

    send_btn.click(respond,       [msg, chatbot],        [chatbot, msg])
    msg.submit(respond,           [msg, chatbot],        [chatbot, msg])
    voice_btn.click(handle_voice, [voice_input, chatbot],[chatbot, msg])
    clear_btn.click(clear_chat,   outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(share=True)
