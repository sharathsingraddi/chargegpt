# ============================================================
# ChargeGPT — app.py
# Full Streamlit Application
# ============================================================

import os
import streamlit as st
import pandas as pd
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="ChargeGPT",
    page_icon="⚡",
    layout="centered"
)

# ============================================================
# SETUP — Load environment and data
# ============================================================
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
# RAG RETRIEVAL
# ============================================================
def retrieve(question, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

# ============================================================
# INTENT DETECTION + ROUTING
# ============================================================
def detect_intent(question):
    system_prompt = """You are a query classifier for an EV charging assistant.
Classify the user's question into exactly ONE of these categories:
- sessions: anything about charging times, peak hours, energy, carbon, duration, utilisation, seasons, busiest times
- drivers: anything about what drivers think, prefer, feel, satisfaction, waiting, how often they charge
- stations: anything about charging stations in Newcastle, locations, postcodes, connectors, fast chargers
- general: anything else about EVs or charging

The question may be casual, slang, or informal — classify the intent not the wording.
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
        """
    else:
        analytics_result = "No specific dataset analytics available for this question."

    rag_context = retrieve(question)
    return intent, analytics_result, rag_context

# ============================================================
# ANSWER FUNCTIONS — 3 Models
# ============================================================
def answer(question):
    intent, analytics_result, rag_context = route(question)
    rag_text = "\n".join(rag_context)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="""You are ChargeGPT, a friendly and knowledgeable EV charging assistant for Newcastle.
Answer in a conversational, natural tone — not robotic or formal.
Use the analytics data and evidence provided to ground your answers in real numbers.
If someone asks casually, respond casually but still give accurate data.
Always include specific numbers from the data.
Never invent numbers not present in the provided data.
Keep answers concise — 3 to 5 sentences maximum.""",
        messages=[{
            "role": "user",
            "content": f"Analytics data:\n{analytics_result}\n\nEvidence:\n{rag_text}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

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
# STREAMLIT UI — Premium Version
# ============================================================

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
}

/* Title styling */
h1 {
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

/* Caption */
.stApp [data-testid="stCaptionContainer"] p {
    color: #8892a4 !important;
    font-size: 0.95rem !important;
}

/* Chat messages */
.stChatMessage {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    backdrop-filter: blur(10px);
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(0, 212, 255, 0.08) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(123, 47, 247, 0.08) !important;
    border-color: rgba(123, 47, 247, 0.2) !important;
}

/* Chat input */
.stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 0.95rem !important;
}

.stChatInput textarea:focus {
    border-color: rgba(0, 212, 255, 0.7) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.15) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #00d4ff !important;
}

/* Markdown text */
.stMarkdown p {
    color: #c8d3e0 !important;
    line-height: 1.7 !important;
}

/* Mode badge */
.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ChargeGPT")
st.caption("Your AI-powered EV charging assistant for Newcastle — ask anything, anytime")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.selectbox(
        "Response mode",
        ["Full ChargeGPT", "LLM + RAG only", "LLM only (baseline)"],
        help="Full ChargeGPT uses real dataset analytics + RAG for maximum accuracy"
    )

    st.markdown("---")

    # Mode explanation
    if mode == "Full ChargeGPT":
        st.markdown("🟢 **Full system active**")
        st.caption("Analytics engine + RAG + Claude LLM. Most accurate.")
    elif mode == "LLM + RAG only":
        st.markdown("🟡 **RAG mode active**")
        st.caption("Knowledge base retrieval + Claude LLM.")
    else:
        st.markdown("🔴 **Baseline mode**")
        st.caption("Claude LLM only. No real data grounding.")

    st.markdown("---")
    st.markdown("### 📊 Data sources")
    st.markdown("**Sessions dataset**")
    st.caption("29,775 real charging sessions")
    st.markdown("**Driver survey**")
    st.caption("124 NE England EV drivers")
    st.markdown("**Stations registry**")
    st.caption("198 Newcastle charging stations")

    st.markdown("---")
    st.markdown("### 💡 Try asking")
    st.caption("When's the busiest time to charge?")
    st.caption("Do most drivers prefer fast chargers?")
    st.caption("Which part of Newcastle has most chargers?")
    st.caption("What season is cleanest for charging?")

# Initialise conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hey! I'm ChargeGPT ⚡ — your EV charging assistant for Newcastle. I'm connected to real charging data so my answers are based on actual numbers, not guesses. What do you want to know?"
    })

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("Ask me anything about EV charging in Newcastle...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Checking the data..."):
            if mode == "Full ChargeGPT":
                response = answer(user_input)
            elif mode == "LLM + RAG only":
                response = answer_rag_only(user_input)
            else:
                response = answer_llm_only(user_input)
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
