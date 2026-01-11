import re
import socket
from typing import List

from faster_whisper import WhisperModel
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


# ==========================
# CONFIG
# ==========================
AUDIO_FILE = "customer.wav"
WHISPER_MODEL = "small"


# ==========================
# ASR MODULE
# ==========================
class ASR:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments)


print("Transcribing customer audio...")
asr = ASR()
raw_text = asr.transcribe(AUDIO_FILE)
print("RAW TRANSCRIPT:", raw_text)


# ==========================
# TEXT CLEANING
# ==========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(uh|um|you know|like)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


cleaned_text = clean_text(raw_text)
print("CLEANED TEXT:", cleaned_text)


# ==========================
# OUTPUT SCHEMA
# ==========================
class IntentOutput(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]


# ==========================
# LLM SETUP
# ==========================
llm = ChatOllama(
    model="tinyllama",
    temperature=0
)

parser = PydanticOutputParser(
    pydantic_object=IntentOutput
)

intent_prompt = ChatPromptTemplate.from_template("""
You are an intent and sentiment classifier for sales calls.

Text:
{text}

Classify and return ONLY valid JSON:
- intent (pricing_objection, interest, complaint, purchase_intent, other)
- sentiment (positive, neutral, negative)
- entities (keywords)

{format_instructions}
""")

intent_chain = intent_prompt | llm | parser


# ==========================
# FALLBACK LOGIC
# ==========================
def fallback_intent(text: str) -> IntentOutput:
    if "price" in text or "cost" in text:
        return IntentOutput(
            intent="pricing_objection",
            sentiment="negative",
            entities=["price"]
        )
    if "buy" in text or "purchase" in text:
        return IntentOutput(
            intent="purchase_intent",
            sentiment="positive",
            entities=["purchase"]
        )
    return IntentOutput(
        intent="other",
        sentiment="neutral",
        entities=[]
    )


# ==========================
# INTENT EXTRACTION
# ==========================
print("Extracting intent and sentiment...")

ollama_reachable = True
try:
    sock = socket.create_connection(("localhost", 11434), timeout=1)
    sock.close()
except Exception:
    ollama_reachable = False

if not ollama_reachable:
    print("Ollama not running — using rule-based fallback.")
    intent_result = fallback_intent(cleaned_text)
else:
    try:
        intent_result = intent_chain.invoke({
            "text": cleaned_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception:
        print("LLM output invalid — switching to fallback.")
        intent_result = fallback_intent(cleaned_text)

print("INTENT RESULT:", intent_result)


# ==========================
# DECISION ENGINE
# ==========================
def decide_action(intent_data: IntentOutput) -> str:
    if intent_data.intent == "pricing_objection":
        if intent_data.sentiment == "negative":
            return "Empathize with concern, then explain ROI before discount"
        return "Explain pricing structure clearly"

    if intent_data.intent == "complaint":
        return "Acknowledge issue and ask clarifying question"

    if intent_data.intent == "purchase_intent":
        return "Move to close and discuss onboarding"

    return "Provide general clarification"


action = decide_action(intent_result)


# ==========================
# SALES RECOMMENDATION
# ==========================
recommendation_prompt = ChatPromptTemplate.from_template("""
You are a sales coach.

Intent: {intent}
Sentiment: {sentiment}
Entities: {entities}

Give ONE short recommendation for the sales agent.
""")

recommendation_chain = recommendation_prompt | llm

recommendation = recommendation_chain.invoke({
    "intent": intent_result.intent,
    "sentiment": intent_result.sentiment,
    "entities": ", ".join(intent_result.entities)
})


# ==========================
# FINAL OUTPUT
# ==========================
print("\n==============================")
print("Customer said        :", raw_text)
print("Detected intent      :", intent_result.intent)
print("Detected sentiment   :", intent_result.sentiment)
print("Sales action         :", action)
print("Sales recommendation :", recommendation.content)
print("==============================")
