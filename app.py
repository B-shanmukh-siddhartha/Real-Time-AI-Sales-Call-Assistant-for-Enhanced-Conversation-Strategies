import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import os

from faster_whisper import WhisperModel

# =============================
# CONFIG
# =============================
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
ENERGY_THRESHOLD = 0.002

# =============================
# WHISPER
# =============================
asr = WhisperModel("small", compute_type="int8")

def transcribe_audio(audio):
    if np.abs(audio).mean() < ENERGY_THRESHOLD:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        segments, _ = asr.transcribe(tmp.name, beam_size=1, max_new_tokens=64)
        return " ".join(seg.text for seg in segments)
    finally:
        os.remove(tmp.name)

# =============================
# RULE-BASED INTENT
# =============================
def decide_action(text):
    t = text.lower()

    if any(w in t for w in ["not happy", "don't like", "bad", "issue", "problem"]):
        return "Acknowledge the concern and empathize"

    if any(w in t for w in ["price", "cost", "pricing", "how much"]):
        return "Explain product details clearly"

    if any(w in t for w in ["suggest", "recommend", "improve"]):
        return "Thank them and note the feedback"

    if any(w in t for w in ["hello", "hi", "thanks", "thank you"]):
        return "Respond politely and build rapport"

    return "Provide general clarification"

# =============================
# UI
# =============================
st.title("🎧 Live Sales Call Recommender")
st.write("Click record → speak → get recommendation")

st.divider()

if st.button("🎤 Record 3 Seconds"):
    with st.spinner("Listening... Speak now"):
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

    text = transcribe_audio(audio)

    if text.strip():
        st.subheader("🗣️ Transcript")
        st.write(text)

        st.subheader("🟢 Sales Recommendation")
        st.success(decide_action(text))
    else:
        st.warning("No clear speech detected. Please try again.")
