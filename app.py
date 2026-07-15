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
# STREAMLIT UI — Professional Design
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* App background — clean dark */
.stApp {
    background: #0b0f19;
}

/* Hide streamlit branding */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Main container width */
.block-container {
    max-width: 780px !important;
    padding-top: 2.5rem !important;
}

/* Title */
h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0 !important;
}

/* Caption under title */
.stApp [data-testid="stCaptionContainer"] p {
    color: #64748b !important;
    font-size: 0.92rem !important;
    margin-top: 4px !important;
}

/* Chat messages — clean cards */
.stChatMessage {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}

/* User messages — subtle accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #0f1b2d !important;
    border-color: #1e3a5f !important;
}

/* Message text */
.stChatMessage p, .stChatMessage li {
    color: #e2e8f0 !important;
    font-size: 0.94rem !important;
    line-height: 1.65 !important;
}

.stChatMessage strong {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Chat input */
.stChatInput {
    padding-bottom: 1rem !important;
}

.stChatInput textarea {
    background: #111827 !important;
    border: 1px solid #2d3b50 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.94rem !important;
    padding: 14px 18px !important;
}

.stChatInput textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

.stChatInput textarea::placeholder {
    color: #475569 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e293b !important;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f1f5f9 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 1.2rem 0 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #2d3b50 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.9rem !important;
}

.stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* Follow-up buttons — pill style */
.stButton button {
    background: transparent !important;
    border: 1px solid #2d3b50 !important;
    border-radius: 100px !important;
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    width: 100% !important;
}

.stButton button:hover {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #3b82f6 !important;
}

/* Map container */
[data-testid="stDeckGlJsonChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #1e293b !important;
}

/* Caption text in sidebar */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #64748b !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0b0f19;
}
::-webkit-scrollbar-thumb {
    background: #2d3b50;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Header with badge
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("⚡ ChargeGPT")
    st.caption("AI-powered EV charging intelligence for Newcastle — grounded in real data")
with col2:
    st.markdown("""
    <div style="text-align:right; padding-top:20px;">
        <span style="background:#052e16; color:#4ade80; padding:5px 12px;
        border-radius:100px; font-size:0.72rem; font-weight:600;
        border:1px solid #14532d;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 12px 0;">
        <span style="font-size:1.4rem; font-weight:700; color:#fff;">⚡ ChargeGPT</span><br>
        <span style="font-size:0.75rem; color:#64748b;">v1.0 — Newcastle University</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### System Mode")
    mode = st.selectbox(
        "Response mode",
        ["Full ChargeGPT", "LLM + RAG only", "LLM only (baseline)"],
        label_visibility="collapsed"
    )

    if mode == "Full ChargeGPT":
        st.markdown("""
        <div style="background:#052e16; border:1px solid #14532d; border-radius:10px; padding:10px 14px; margin-top:8px;">
            <span style="color:#4ade80; font-size:0.8rem; font-weight:600;">● Full system active</span><br>
            <span style="color:#64748b; font-size:0.75rem;">Analytics engine + RAG + Claude LLM</span>
        </div>
        """, unsafe_allow_html=True)
    elif mode == "LLM + RAG only":
        st.markdown("""
        <div style="background:#2d2006; border:1px solid #713f12; border-radius:10px; padding:10px 14px; margin-top:8px;">
            <span style="color:#facc15; font-size:0.8rem; font-weight:600;">● RAG mode active</span><br>
            <span style="color:#64748b; font-size:0.75rem;">Knowledge base + Claude LLM</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2d0a0a; border:1px solid #7f1d1d; border-radius:10px; padding:10px 14px; margin-top:8px;">
            <span style="color:#f87171; font-size:0.8rem; font-weight:600;">● Baseline mode</span><br>
            <span style="color:#64748b; font-size:0.75rem;">No data grounding — may hallucinate</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    <div style="font-size:0.82rem; color:#94a3b8; line-height:2;">
    <span style="color:#3b82f6;">▸</span> 29,775 charging sessions<br>
    <span style="color:#3b82f6;">▸</span> 124 EV driver surveys<br>
    <span style="color:#3b82f6;">▸</span> 198 Newcastle stations
    </div>
    <div style="font-size:0.72rem; color:#475569; margin-top:6px;">All data verified · July 2026</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Prompts")
    st.markdown("""
    <div style="font-size:0.8rem; color:#94a3b8; line-height:2.1;">
    "Nearest charger to NE4 6PL"<br>
    "When's the busiest time to charge?"<br>
    "Where should new stations go?"<br>
    "Do drivers prefer fast chargers?"
    </div>
    """, unsafe_allow_html=True)

# Initialise state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hey! I'm **ChargeGPT** — your EV charging assistant for Newcastle.\n\nI can find your nearest charger, analyse demand patterns, and advise planners on where to build next. Everything I say is grounded in real data.\n\nWhat would you like to know?"
    })

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "last_intent" not in st.session_state:
    st.session_state.last_intent = "general"

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Follow-up buttons
if len(st.session_state.messages) > 1:
    suggestions = FOLLOW_UPS.get(st.session_state.last_intent, FOLLOW_UPS["general"])
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"followup_{len(st.session_state.messages)}_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

# Input
user_input = st.chat_input("Ask anything about EV charging in Newcastle...")

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
        with st.spinner("Analysing real data..."):
            if mode == "Full ChargeGPT":
                response, intent = answer(question_to_process)
                st.session_state.last_intent = intent
                response = response + "\n\n" + SOURCE_LABELS.get(intent, SOURCE_LABELS["general"])
            elif mode == "LLM + RAG only":
                response = answer_rag_only(question_to_process)
                intent = None
            else:
                response = answer_llm_only(question_to_process)
                intent = None
        st.markdown(response)

        if intent in ["stations", "nearest", "planning"]:
            st.map(stations[["latitude", "longitude"]])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.rerun()
