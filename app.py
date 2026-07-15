# ============================================================
# ChargeGPT — app.py
# Complete version: memory + source labels + follow-ups
# ============================================================

import os
import math
import requests
import streamlit as st
import pandas as pd
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

st.set_page_config(
    page_title="ChargeGPT",
    page_icon="⚡",
    layout="centered"
)

# ============================================================
# SETUP
# ============================================================
load_dotenv()

try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

@st.cache_data
def load_data():
    usb = pd.read_csv("usb_features.csv")
    usb["sessionStart"] = pd.to_datetime(usb["sessionStart"])
    usb["sessionStop"] = pd.to_datetime(usb["sessionStop"])
    drivers = pd.read_csv("drivers_cleaned.csv")
    stations = pd.read_csv("stations_cleaned.csv")
    return usb, drivers, stations

@st.cache_resource
def setup_rag():
    with open("knowledge_base.txt", "r") as file:
        content = file.read()
    all_lines = content.split("\n")
    chunks = [line.strip() for line in all_lines
              if line.strip() != "" and not line.startswith("===")]
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="chargegpt_knowledge")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

usb, drivers, stations = load_data()
collection = setup_rag()

# ============================================================
# ANALYTICS FUNCTIONS — Sessions
# ============================================================
def peak_hours(df):
    result = df.groupby("hour").size().reset_index()
    result.columns = ["hour", "session_count"]
    return result.sort_values("session_count", ascending=False)

def avg_energy_per_session(df):
    return df["energy_consumption(kWh)"].mean()

def avg_duration(df):
    return df["duration_hrs"].mean()

def seasonal_demand(df):
    result = df.groupby("Season")["energy_consumption(kWh)"].mean().reset_index()
    result.columns = ["season", "avg_energy"]
    return result.sort_values("avg_energy", ascending=False)

def top_utilisation_chargers(df):
    result = df.groupby("chargerId")["time_based_util_rate"].mean().reset_index()
    result.columns = ["chargerId", "avg_utilisation"]
    return result.sort_values("avg_utilisation", ascending=False)

def carbon_by_season(df):
    result = df.groupby("Season")["Carbon_Emissions_(gCO2)"].mean().reset_index()
    result.columns = ["season", "carbon_emissions"]
    return result.sort_values("carbon_emissions", ascending=False)

# ============================================================
# ANALYTICS FUNCTIONS — Drivers
# ============================================================
def preferred_charge_time(df):
    result = df.groupby("preferred_charge_time").size().reset_index()
    result.columns = ["time_slot", "driver_count"]
    return result.sort_values("driver_count", ascending=False)

def charger_preference(df):
    result = df.groupby("ac_vs_dc_preference").size().reset_index()
    result.columns = ["preference", "drivers_count"]
    return result.sort_values("drivers_count", ascending=False)

def avg_satisfaction(df):
    return df["satisfaction_score"].mean()

# ============================================================
# ANALYTICS FUNCTIONS — Stations
# ============================================================
def stations_by_postcode(df):
    result = df.groupby("postcode").size().reset_index()
    result.columns = ["postcode", "station_count"]
    return result.sort_values("station_count", ascending=False)

def fast_charger_count(df):
    result = df.groupby("connector1Type").size().reset_index()
    result.columns = ["connector_type", "station_count"]
    return result.sort_values("station_count", ascending=False)

def chargers_open_24hr(df):
    result = df.groupby("access24Hours").size().reset_index()
    result.columns = ["access24Hours", "count"]
    return result

# ============================================================
# NEAREST STATION FINDER
# ============================================================
def postcode_to_coords(postcode):
    try:
        url = f"https://api.postcodes.io/postcodes/{postcode.replace(' ', '')}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data["status"] == 200:
            return data["result"]["latitude"], data["result"]["longitude"]
        return None, None
    except Exception:
        return None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def find_nearest_stations(postcode, n=3):
    lat, lon = postcode_to_coords(postcode)
    if lat is None:
        return None

    df = stations.copy()
    df["distance_km"] = df.apply(
        lambda row: haversine_distance(lat, lon, row["latitude"], row["longitude"]),
        axis=1
    )
    nearest = df.sort_values("distance_km").head(n)

    results = []
    for _, row in nearest.iterrows():
        results.append(f"""
Station: {row['name']}
Distance: {row['distance_km']:.2f} km
Postcode: {row['postcode']}
Connector: {row['connector1Type']}
Power: {row['connector1RatedOutputKW']} kW
Charge method: {row['connector1ChargeMethod']}
Payment required: {row['paymentRequired']}
24 hour access: {row['access24Hours']}
Status: {row['chargeDeviceStatus']}
""")
    return "\n".join(results)

# ============================================================
# INFRASTRUCTURE GAP ANALYSIS
# ============================================================
def infrastructure_gap_analysis():
    station_counts = stations_by_postcode(stations)
    low_supply = station_counts.tail(10)

    if "pc" in drivers.columns:
        driver_postcodes = drivers["pc"].value_counts().reset_index()
    else:
        driver_postcodes = drivers["postcode"].value_counts().reset_index()
    driver_postcodes.columns = ["postcode", "driver_count"]

    dc_count = len(stations[stations["connector1ChargeMethod"] == "DC"])
    avg_power = stations["connector1RatedOutputKW"].mean()

    return f"""
INFRASTRUCTURE GAP ANALYSIS:

Postcodes with FEWEST stations (potential build locations):
{low_supply.to_string()}

Postcode districts where surveyed drivers live (demand indicators):
{driver_postcodes.head(10).to_string()}

Key gaps:
- Only {dc_count} DC fast chargers across all 198 stations
- Average power output is only {avg_power:.1f} kW
- 79% of drivers prefer DC fast charging but only 3.5% of stations provide it
- Stations concentrated in NE1 city centre while outer areas underserved
- Driver satisfaction is 2.77/5 and 66% report difficulty finding stations
"""

# ============================================================
# RAG RETRIEVAL
# ============================================================
def retrieve(question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

# ============================================================
# INTENT DETECTION + POSTCODE EXTRACTION
# ============================================================
def detect_intent(question):
    system_prompt = """You are a query classifier for an EV charging assistant.
Classify the user's question into exactly ONE of these categories:
- sessions: anything about charging times, peak hours, energy, carbon, duration, utilisation, seasons, busiest times
- drivers: anything about what drivers think, prefer, feel, satisfaction, waiting, how often they charge
- stations: anything about charging stations in Newcastle generally, counts, connector types, 24 hour access
- nearest: user gives a postcode or location and wants to find the nearest charging station
- planning: questions about where to BUILD new stations, infrastructure gaps, underserved areas, city planner recommendations
- general: anything else about EVs or charging

The question may be casual, slang, or informal — classify the intent not the wording.
Consider the conversation context if provided.
Reply with ONLY the category name — one word, lowercase, no punctuation."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Classify this question: {question}"
        }]
    )
    return response.content[0].text.strip().lower()

def extract_postcode(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=15,
        system="Extract the UK postcode from the message. Reply with ONLY the postcode, nothing else. If no postcode found reply NONE.",
        messages=[{"role": "user", "content": question}]
    )
    postcode = response.content[0].text.strip()
    return None if postcode == "NONE" else postcode

# ============================================================
# SOURCE LABELS
# ============================================================
SOURCE_LABELS = {
    "sessions": "📊 *Source: EV Charging Sessions Dataset (29,775 verified sessions)*",
    "drivers": "📊 *Source: NE England EV Driver Survey (124 respondents)*",
    "stations": "📊 *Source: Newcastle Charging Stations Registry (198 stations)*",
    "nearest": "📊 *Source: Newcastle Charging Stations Registry + postcodes.io geolocation*",
    "planning": "📊 *Source: Cross-analysis of all 3 datasets (sessions + survey + stations)*",
    "general": "📊 *Source: ChargeGPT knowledge base*"
}

# ============================================================
# ROUTING
# ============================================================
def route(question):
    intent = detect_intent(question)

    if intent == "sessions":
        analytics_result = f"""
        Peak hour: {peak_hours(usb).iloc[0]['hour']}:00 with {peak_hours(usb).iloc[0]['session_count']} sessions
        Avg energy: {avg_energy_per_session(usb):.2f} kWh
        Avg duration: {avg_duration(usb):.2f} hours
        Carbon by season: {carbon_by_season(usb).to_string()}
        Seasonal demand: {seasonal_demand(usb).to_string()}
        """
    elif intent == "drivers":
        analytics_result = f"""
        Charger preference: {charger_preference(drivers).to_string()}
        Preferred charge time: {preferred_charge_time(drivers).to_string()}
        Avg satisfaction: {avg_satisfaction(drivers):.2f} out of 5
        """
    elif intent == "stations":
        analytics_result = f"""
        Stations by postcode: {stations_by_postcode(stations).head(5).to_string()}
        Connector types: {fast_charger_count(stations).to_string()}
        24hr access: {chargers_open_24hr(stations).to_string()}
        Total stations: {len(stations)}
        """
    elif intent == "nearest":
        postcode = extract_postcode(question)
        if postcode:
            nearest_info = find_nearest_stations(postcode)
            if nearest_info:
                analytics_result = f"NEAREST STATIONS TO {postcode}:\n{nearest_info}"
            else:
                analytics_result = f"Could not find coordinates for postcode {postcode}. It may be invalid."
        else:
            analytics_result = "No postcode detected. Ask the user to provide their postcode."
    elif intent == "planning":
        analytics_result = infrastructure_gap_analysis()
    else:
        analytics_result = "No specific dataset analytics available for this question."

    rag_context = retrieve(question)
    return intent, analytics_result, rag_context

# ============================================================
# ANSWER FUNCTIONS — with conversation memory
# ============================================================
def build_conversation_context():
    """Get last 4 messages for context"""
    if "messages" not in st.session_state:
        return ""
    recent = st.session_state.messages[-4:]
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(context_lines)

def answer(question):
    conversation_context = build_conversation_context()
    intent, analytics_result, rag_context = route(question)
    rag_text = "\n".join(rag_context)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system="""You are ChargeGPT, a friendly and knowledgeable EV charging assistant for Newcastle.
Answer in a conversational, natural tone — not robotic or formal.
Use the analytics data and evidence provided to ground your answers in real numbers.
Use the conversation history to understand follow-up questions and context.
If someone asks casually, respond casually but still give accurate data.
For nearest station queries, list stations clearly with distance and key features.
For planning queries, give clear recommendations based on the gap analysis.
Always include specific numbers from the data.
Never invent numbers not present in the provided data.
Keep answers concise.""",
        messages=[{
            "role": "user",
            "content": f"""Conversation history:
{conversation_context}

Analytics data:
{analytics_result}

Evidence:
{rag_text}

Current question: {question}"""
        }]
    )
    return response.content[0].text, intent

def answer_llm_only(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

def answer_rag_only(question):
    context = retrieve(question)
    rag_text = "\n".join(context)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="Answer using only the evidence provided. Quote specific numbers.",
        messages=[{
            "role": "user",
            "content": f"Evidence:\n{rag_text}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

# ============================================================
# FOLLOW-UP SUGGESTIONS
# ============================================================
FOLLOW_UPS = {
    "sessions": [
        "Which season is cleanest for charging?",
        "What's the quietest time to charge?",
        "How long do sessions usually last?"
    ],
    "drivers": [
        "How satisfied are drivers with charging?",
        "How long will drivers wait for a charger?",
        "How often do drivers charge weekly?"
    ],
    "stations": [
        "How many fast chargers exist in Newcastle?",
        "Which stations are open 24 hours?",
        "Which postcode has the most chargers?"
    ],
    "nearest": [
        "Are any of these open 24 hours?",
        "Which one is a fast charger?",
        "Show stations near NE1 instead"
    ],
    "planning": [
        "Which postcodes need DC fast chargers most?",
        "What do drivers say about availability?",
        "Where is demand highest?"
    ],
    "general": [
        "What are the peak charging hours?",
        "Find my nearest charging station",
        "Where should new stations be built?"
    ]
}

# ============================================================
# STREAMLIT UI
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
}
h1 {
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}
.stApp [data-testid="stCaptionContainer"] p {
    color: #8892a4 !important;
}
.stChatMessage {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(0, 212, 255, 0.08) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(123, 47, 247, 0.08) !important;
    border-color: rgba(123, 47, 247, 0.2) !important;
}
.stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 12px !important;
    color: white !important;
}
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
.stMarkdown p {
    color: #c8d3e0 !important;
    line-height: 1.7 !important;
}
.stButton button {
    background: rgba(0, 212, 255, 0.08) !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 20px !important;
    color: #8fd8ff !important;
    font-size: 0.8rem !important;
    padding: 4px 14px !important;
}
.stButton button:hover {
    background: rgba(0, 212, 255, 0.18) !important;
    border-color: rgba(0, 212, 255, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ChargeGPT")
st.caption("Your AI-powered EV charging assistant for Newcastle — for drivers and city planners")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.selectbox(
        "Response mode",
        ["Full ChargeGPT", "LLM + RAG only", "LLM only (baseline)"],
        help="Full ChargeGPT uses real dataset analytics + RAG for maximum accuracy"
    )

    st.markdown("---")
    if mode == "Full ChargeGPT":
        st.markdown("🟢 **Full system active**")
        st.caption("Analytics + RAG + Claude LLM")
    elif mode == "LLM + RAG only":
        st.markdown("🟡 **RAG mode active**")
        st.caption("Knowledge base + Claude LLM")
    else:
        st.markdown("🔴 **Baseline mode**")
        st.caption("Claude LLM only. No data grounding.")

    st.markdown("---")
    st.markdown("### 📊 Data sources")
    st.caption("• 29,775 verified charging sessions")
    st.caption("• 124 NE England driver surveys")
    st.caption("• 198 Newcastle charging stations")
    st.caption("Last updated: July 2026")

    st.markdown("---")
    st.markdown("### 💡 Try asking")
    st.caption("🚗 Nearest charger to NE4 6PL?")
    st.caption("📈 When's the busiest time to charge?")
    st.caption("🏗️ Where should new stations be built?")
    st.caption("🔌 Do drivers prefer fast chargers?")

# Initialise state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hey! I'm ChargeGPT ⚡ — your EV charging assistant for Newcastle. I can find your nearest charger, show demand patterns, and advise city planners on where to build next. All grounded in real data. What do you want to know?"
    })

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "last_intent" not in st.session_state:
    st.session_state.last_intent = "general"

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Follow-up suggestion buttons
if len(st.session_state.messages) > 1:
    suggestions = FOLLOW_UPS.get(st.session_state.last_intent, FOLLOW_UPS["general"])
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"followup_{len(st.session_state.messages)}_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

# Get user input
user_input = st.chat_input("Ask me anything — try 'nearest charger to NE4 6PL'")

# Handle either typed input or clicked follow-up
question_to_process = None
if user_input:
    question_to_process = user_input
elif st.session_state.pending_question:
    question_to_process = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_process:
    with st.chat_message("user"):
        st.markdown(question_to_process)
    st.session_state.messages.append({"role": "user", "content": question_to_process})

    with st.chat_message("assistant"):
        with st.spinner("Checking the data..."):
            if mode == "Full ChargeGPT":
                response, intent = answer(question_to_process)
                st.session_state.last_intent = intent
                # Add source label
                response = response + "\n\n" + SOURCE_LABELS.get(intent, SOURCE_LABELS["general"])
            elif mode == "LLM + RAG only":
                response = answer_rag_only(question_to_process)
                intent = None
            else:
                response = answer_llm_only(question_to_process)
                intent = None
        st.markdown(response)

        # Show map for station-related queries
        if intent in ["stations", "nearest", "planning"]:
            st.map(stations[["latitude", "longitude"]])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.rerun()
