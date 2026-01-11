import sounddevice as sd
import numpy as np
import queue
import tempfile
import soundfile as sf
import os
import json
import time

from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# =============================
# CONFIG
# =============================
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3
ENERGY_THRESHOLD = 0.002

audio_queue = queue.Queue()
buffer = np.zeros((0, 1))

print("Program started")

# =============================
# AUDIO CALLBACK
# =============================
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# Keep stream alive (important on Windows)
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    device=None,
    callback=audio_callback
)
stream.start()

# =============================
# WHISPER (SAFE MODE)
# =============================
asr = WhisperModel("small", compute_type="int8")

def get_audio_chunk():
    global buffer
    while not audio_queue.empty():
        buffer = np.vstack([buffer, audio_queue.get()])

    if len(buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
        chunk = buffer[: SAMPLE_RATE * BUFFER_SECONDS]
        buffer = buffer[SAMPLE_RATE * BUFFER_SECONDS :]
        return chunk
    return None


def transcribe_chunk(audio):
    # Skip silence
    if np.abs(audio).mean() < ENERGY_THRESHOLD:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        segments, _ = asr.transcribe(
            tmp.name,
            beam_size=1,
            max_new_tokens=64
        )
        return " ".join(seg.text for seg in segments)
    except Exception:
        return ""
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

# =============================
# RULE-BASED INTENT (PRIMARY)
# =============================
def rule_based_intent(text: str):
    text = text.lower()

    if any(w in text for w in ["not happy", "don't like", "dont like", "bad", "worst", "issue", "problem"]):
        return {"intent": "complaint", "sentiment": "bad"}

    if any(w in text for w in ["price", "cost", "pricing", "how much", "charges"]):
        return {"intent": "inquiry", "sentiment": "neutral"}

    if any(w in text for w in ["suggest", "recommend", "improve", "feedback"]):
        return {"intent": "suggestion", "sentiment": "neutral"}

    if any(w in text for w in ["hello", "hi", "thank you", "thanks"]):
        return {"intent": "greeting", "sentiment": "good"}

    return None

# =============================
# LLM (FALLBACK ONLY)
# =============================
prompt = ChatPromptTemplate.from_template("""
Classify the customer text.

Text:
{text}

Return ONLY JSON:
{{"intent":"greeting|inquiry|complaint|suggestion|other",
  "sentiment":"good|bad|neutral"}}
""")

llm = ChatOllama(model="tinyllama", temperature=0)
chain = prompt | llm


def get_intent(text: str):
    # 1️⃣ Rule-based first (FAST & reliable)
    rule_result = rule_based_intent(text)
    if rule_result:
        return rule_result

    # 2️⃣ Skip LLM for very short text
    if len(text.split()) <= 2:
        return {"intent": "greeting", "sentiment": "neutral"}

    # 3️⃣ LLM fallback
    try:
        resp = chain.invoke({"text": text}, config={"timeout": 6})
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return json.loads(raw)
    except Exception:
        return {"intent": "other", "sentiment": "neutral"}

# =============================
# BUSINESS LOGIC
# =============================
def decide_action(intent: str):
    if intent == "complaint":
        return "Acknowledge the concern and empathize"
    if intent == "inquiry":
        return "Explain product details clearly"
    if intent == "greeting":
        return "Respond politely and build rapport"
    if intent == "suggestion":
        return "Thank them and note the feedback"
    return "Provide general clarification"

# =============================
# MAIN LOOP
# =============================
print("🎤 Listening… Speak ONE sentence and STOP\n")

try:
    while True:
        chunk = get_audio_chunk()
        if chunk is None:
            time.sleep(0.05)
            continue

        text = transcribe_chunk(chunk)
        if not text.strip():
            continue

        print("Customer:", text)

        result = get_intent(text.lower())
        action = decide_action(result["intent"])

        print("🟢 SALES TIP:", action)
        print("-" * 40)

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop()
    stream.close()
