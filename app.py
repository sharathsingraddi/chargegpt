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
- sessions: questions about charging sessions, peak hours, energy consumption, carbon emissions, utilisation rates, seasonal patterns
- drivers: questions about driver behaviour, preferences, satisfaction scores, wait times, charging frequency, charger type preference
- stations: questions about Newcastle charging stations, locations, postcodes, connector types, power output, 24 hour access
- general: any other general question about EVs or charging
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
        system="""You are ChargeGPT, a data-grounded EV charging assistant for Newcastle.
Answer questions using ONLY the analytics data and evidence provided.
Always quote specific numbers from the data.
Be clear, concise, and helpful.
Never invent numbers not present in the provided data.""",
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
# STREAMLIT UI
# ============================================================
st.title("⚡ ChargeGPT")
st.caption("A data-grounded conversational AI assistant for EV charging infrastructure in Newcastle")

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Select mode",
        ["Full ChargeGPT", "LLM + RAG only", "LLM only (baseline)"],
        help="Choose which system configuration to use"
    )
    st.markdown("---")
    st.markdown("**About ChargeGPT**")
    st.markdown("Built on 3 real Newcastle datasets:")
    st.markdown("- 29,775 charging sessions")
    st.markdown("- 124 EV driver survey responses")
    st.markdown("- 198 Newcastle charging stations")

# Initialise conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am ChargeGPT, your EV charging assistant for Newcastle. Ask me anything about charging sessions, driver behaviour, or station infrastructure."
    })

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("Ask me about EV charging in Newcastle...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analysing data..."):
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