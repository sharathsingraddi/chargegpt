# ============================================================
# ChargeGPT — app.py — FINAL
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
from datetime import datetime

st.set_page_config(page_title="ChargeGPT", page_icon="⚡", layout="centered")

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
    chunks = [line.strip() for line in content.split("\n")
              if line.strip() != "" and not line.startswith("===")]
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="chargegpt_knowledge")
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return collection

usb, drivers, stations = load_data()
collection = setup_rag()

# ============================================================
# ANALYTICS FUNCTIONS
# ============================================================
def peak_hours(df):
    r = df.groupby("hour").size().reset_index()
    r.columns = ["hour", "session_count"]
    return r.sort_values("session_count", ascending=False)

def avg_energy_per_session(df):
    return df["energy_consumption(kWh)"].mean()

def avg_duration(df):
    return df["duration_hrs"].mean()

def seasonal_demand(df):
    r = df.groupby("Season")["energy_consumption(kWh)"].mean().reset_index()
    r.columns = ["season", "avg_energy"]
    return r.sort_values("avg_energy", ascending=False)

def top_utilisation_chargers(df):
    r = df.groupby("chargerId")["time_based_util_rate"].mean().reset_index()
    r.columns = ["chargerId", "avg_utilisation"]
    return r.sort_values("avg_utilisation", ascending=False)

def carbon_by_season(df):
    r = df.groupby("Season")["Carbon_Emissions_(gCO2)"].mean().reset_index()
    r.columns = ["season", "carbon_emissions"]
    return r.sort_values("carbon_emissions", ascending=False)

def preferred_charge_time(df):
    r = df.groupby("preferred_charge_time").size().reset_index()
    r.columns = ["time_slot", "driver_count"]
    return r.sort_values("driver_count", ascending=False)

def charger_preference(df):
    r = df.groupby("ac_vs_dc_preference").size().reset_index()
    r.columns = ["preference", "drivers_count"]
    return r.sort_values("drivers_count", ascending=False)

def avg_satisfaction(df):
    return df["satisfaction_score"].mean()

def stations_by_postcode(df):
    r = df.groupby("postcode").size().reset_index()
    r.columns = ["postcode", "station_count"]
    return r.sort_values("station_count", ascending=False)

def fast_charger_count(df):
    r = df.groupby("connector1Type").size().reset_index()
    r.columns = ["connector_type", "station_count"]
    return r.sort_values("station_count", ascending=False)

def chargers_open_24hr(df):
    r = df.groupby("access24Hours").size().reset_index()
    r.columns = ["access24Hours", "count"]
    return r

# ============================================================
# LOCATION SEARCH
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

def placename_to_coords(place):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{place}, Newcastle upon Tyne, UK", "format": "json", "limit": 1}
        headers = {"User-Agent": "ChargeGPT-Dissertation-Project"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
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

def find_nearest_stations(lat, lon, n=3):
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

def extract_location(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=30,
        system="""Extract the location from the message. It could be a UK postcode (like NE4 6PL) or a place name (like Primark, Eldon Square, St James Park).
If postcode reply: POSTCODE|<the postcode>
If place name reply: PLACE|<the place name>
If none reply: NONE""",
        messages=[{"role": "user", "content": question}]
    )
    result = response.content[0].text.strip()
    if result == "NONE":
        return None, None
    parts = result.split("|")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

# ============================================================
# GAP ANALYSIS
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
Postcodes with FEWEST stations: {low_supply.to_string()}
Driver postcode districts (demand): {driver_postcodes.head(10).to_string()}
Key gaps:
- Only {dc_count} DC fast chargers across 198 stations
- Average power only {avg_power:.1f} kW
- 79% of drivers prefer DC but only 3.5% of stations provide it
- Stations concentrated in NE1, outer areas underserved
- Satisfaction 2.77/5, 66% report difficulty finding stations
"""

# ============================================================
# RAG + INTENT
# ============================================================
def retrieve(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0]

def detect_intent(question):
    system_prompt = """You are a query classifier for an EV charging assistant.
Classify into exactly ONE category:
- sessions: charging times, peak hours, energy, carbon, duration, utilisation, seasons
- drivers: driver preferences, satisfaction, waiting, charging frequency
- stations: Newcastle stations generally, counts, connector types, 24hr access
- nearest: user gives a postcode OR place name wanting nearest charging station
- planning: where to BUILD new stations, gaps, underserved areas
- general: anything else
Casual wording is fine — classify the intent.
Reply ONLY the category name — one word, lowercase."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Classify: {question}"}]
    )
    return response.content[0].text.strip().lower()

SOURCE_LABELS = {
    "sessions": "*Source: EV Charging Sessions Dataset — 29,775 verified sessions*",
    "drivers": "*Source: NE England EV Driver Survey — 124 respondents*",
    "stations": "*Source: Newcastle Charging Stations Registry — 198 stations*",
    "nearest": "*Source: Stations Registry + OpenStreetMap geolocation*",
    "planning": "*Source: Cross-analysis of all 3 datasets*",
    "general": "*Source: ChargeGPT knowledge base*"
}

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
        loc_type, location = extract_location(question)
        if location:
            if loc_type == "POSTCODE":
                lat, lon = postcode_to_coords(location)
            else:
                lat, lon = placename_to_coords(location)
            if lat is not None:
                analytics_result = f"NEAREST STATIONS TO {location}:\n{find_nearest_stations(lat, lon)}"
            else:
                analytics_result = f"Could not locate '{location}'."
        else:
            analytics_result = "No location detected. Ask user for postcode or landmark."
    elif intent == "planning":
        analytics_result = infrastructure_gap_analysis()
    else:
        analytics_result = "No specific dataset analytics available."
    rag_context = retrieve(question)
    return intent, analytics_result, rag_context

# ============================================================
# ANSWER FUNCTIONS
# ============================================================
def build_conversation_context():
    if "messages" not in st.session_state:
        return ""
    recent = st.session_state.messages[-4:]
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

def answer(question):
    conversation_context = build_conversation_context()
    intent, analytics_result, rag_context = route(question)
    rag_text = "\n".join(rag_context)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system="""You are ChargeGPT, a friendly EV charging assistant for Newcastle.
Answer conversationally, grounded in the provided data.
Use conversation history for follow-up context.
For nearest station queries, list stations clearly with distance and features.
For planning queries, give clear recommendations.
Always quote specific numbers. Never invent numbers. Keep answers concise.""",
        messages=[{"role": "user", "content": f"""Conversation history:
{conversation_context}

Analytics data:
{analytics_result}

Evidence:
{rag_text}

Current question: {question}"""}]
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
    rag_text = "\n".join(retrieve(question))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="Answer using only the evidence provided. Quote specific numbers.",
        messages=[{"role": "user", "content": f"Evidence:\n{rag_text}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

FOLLOW_UPS = {
    "sessions": ["Which season is cleanest?", "Quietest time to charge?", "Average session length?"],
    "drivers": ["How satisfied are drivers?", "How long will drivers wait?", "Weekly charging frequency?"],
    "stations": ["How many fast chargers?", "Which are open 24 hours?", "Most chargers by postcode?"],
    "nearest": ["Any open 24 hours?", "Which is fastest?", "Stations near Eldon Square?"],
    "planning": ["Where are DC chargers needed?", "What do drivers say?", "Where is demand highest?"],
    "general": ["Peak charging hours?", "Find my nearest station", "Where to build new stations?"]
}

# ============================================================
# STATE + HISTORY
# ============================================================
def init_state():
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    if "current_chat_index" not in st.session_state:
        st.session_state.current_chat_index = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = "general"

def welcome_message():
    return {
        "role": "assistant",
        "content": "Hey! I'm **ChargeGPT** — your EV charging assistant for Newcastle.\n\nAsk me anything: find your nearest charger, explore demand patterns, or get planning recommendations. Everything is grounded in real data."
    }

def save_current_chat():
    email = st.session_state.user_email
    if not email:
        return
    if st.session_state.messages and len(st.session_state.messages) > 1:
        if st.session_state.current_chat_index is not None:
            st.session_state.all_chats[email][st.session_state.current_chat_index]["messages"] = st.session_state.messages
        else:
            first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Chat")
            title = first_user_msg[:38] + ("..." if len(first_user_msg) > 38 else "")
            st.session_state.all_chats[email].append({
                "title": title,
                "messages": st.session_state.messages,
                "created": datetime.now().strftime("%d %b %H:%M")
            })
            st.session_state.current_chat_index = len(st.session_state.all_chats[email]) - 1

def new_chat():
    save_current_chat()
    st.session_state.messages = [welcome_message()]
    st.session_state.current_chat_index = None
    st.session_state.last_intent = "general"

def load_chat(index):
    save_current_chat()
    email = st.session_state.user_email
    st.session_state.messages = st.session_state.all_chats[email][index]["messages"]
    st.session_state.current_chat_index = index

init_state()

# ============================================================
# UI — MINIMAL DARK
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', -apple-system, sans-serif !important; }
.stApp { background: #0e0e10; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 760px !important; padding-top: 2rem !important; }

h1 { color: #fafafa !important; font-size: 1.9rem !important; font-weight: 600 !important; letter-spacing: -0.4px !important; margin-bottom: 0 !important; }
.stApp [data-testid="stCaptionContainer"] p { color: #71717a !important; font-size: 0.88rem !important; }

.stChatMessage { background: #17171a !important; border: 1px solid #26262b !important; border-radius: 12px !important; padding: 14px 18px !important; margin-bottom: 10px !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #1c1c21 !important; }
.stChatMessage p, .stChatMessage li { color: #d4d4d8 !important; font-size: 0.92rem !important; line-height: 1.65 !important; }
.stChatMessage strong { color: #fafafa !important; }
.stChatMessage em { color: #71717a !important; font-size: 0.8rem !important; }

.stChatInput textarea { background: #17171a !important; border: 1px solid #303036 !important; border-radius: 12px !important; color: #fafafa !important; font-size: 0.92rem !important; }
.stChatInput textarea:focus { border-color: #52525b !important; box-shadow: none !important; }
.stChatInput textarea::placeholder { color: #52525b !important; }

[data-testid="stSidebar"] { background: #111113 !important; border-right: 1px solid #26262b !important; }
[data-testid="stSidebar"] hr { border-color: #26262b !important; margin: 1rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #52525b !important; font-size: 0.76rem !important; }

.stSelectbox > div > div { background: #17171a !important; border: 1px solid #303036 !important; border-radius: 8px !important; color: #fafafa !important; font-size: 0.86rem !important; }

.stTextInput input { background: #17171a !important; border: 1px solid #303036 !important; border-radius: 8px !important; color: #fafafa !important; font-size: 0.86rem !important; }

/* New chat — white pill like Claude */
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {
    background: #fafafa !important; color: #0e0e10 !important;
    border-radius: 100px !important; font-weight: 600 !important;
    text-align: center !important; border: none !important;
    font-size: 0.86rem !important; padding: 8px 14px !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button:hover { background: #d4d4d8 !important; }

/* History rows — quiet like Claude recents */
[data-testid="stSidebar"] .stButton button {
    background: transparent !important; border: none !important;
    color: #d4d4d8 !important; font-size: 0.84rem !important;
    text-align: left !important; padding: 7px 10px !important;
    border-radius: 8px !important; width: 100% !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton button:hover { background: #1c1c21 !important; color: #fafafa !important; }

/* Main area follow-up buttons */
.block-container .stButton button {
    background: #17171a !important; border: 1px solid #303036 !important;
    border-radius: 100px !important; color: #a1a1aa !important;
    font-size: 0.78rem !important; padding: 5px 12px !important;
    text-align: center !important;
}
.block-container .stButton button:hover { background: #26262b !important; color: #fafafa !important; }

.stSpinner > div { border-top-color: #a1a1aa !important; }
[data-testid="stDeckGlJsonChart"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #26262b !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0e0e10; }
::-webkit-scrollbar-thumb { background: #303036; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ ChargeGPT")
st.caption("Data-grounded EV charging intelligence for Newcastle")

# ============================================================
# SIDEBAR — Claude style
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;">
        <span style="font-size:1.4rem; font-weight:700; color:#fafafa;">⚡ ChargeGPT</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_email is None:
        st.markdown('<div style="font-size:0.82rem; color:#a1a1aa; margin-bottom:6px;">Sign in to save your chats</div>', unsafe_allow_html=True)
        email_input = st.text_input("Email", placeholder="you@email.com", label_visibility="collapsed")
        if email_input and "@" in email_input:
            st.session_state.user_email = email_input.strip().lower()
            if st.session_state.user_email not in st.session_state.all_chats:
                st.session_state.all_chats[st.session_state.user_email] = []
            st.rerun()
    else:
        if st.button("＋  New chat", key="newchat", use_container_width=True):
            new_chat()
            st.rerun()

        user_chats = st.session_state.all_chats.get(st.session_state.user_email, [])
        if user_chats:
            st.markdown("""
            <div style="font-size:0.7rem; color:#71717a; font-weight:600;
            text-transform:uppercase; letter-spacing:1px; padding:14px 0 4px 0;">Recents</div>
            """, unsafe_allow_html=True)
            for i, chat in enumerate(reversed(user_chats)):
                real_index = len(user_chats) - 1 - i
                if st.button(chat["title"], key=f"hist_{real_index}", use_container_width=True):
                    load_chat(real_index)
                    st.rerun()

        st.caption(f"Signed in: {st.session_state.user_email}")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#71717a; font-weight:600;
    text-transform:uppercase; letter-spacing:1px; padding-bottom:4px;">Mode</div>
    """, unsafe_allow_html=True)
    mode = st.selectbox(
        "Response mode",
        ["Full ChargeGPT", "LLM + RAG only", "LLM only (baseline)"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="font-size:0.72rem; color:#52525b; padding-top:12px; line-height:1.8;">
    29,775 sessions · 124 drivers · 198 stations<br>
    Newcastle University · v1.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CHAT AREA
# ============================================================
if not st.session_state.messages:
    st.session_state.messages = [welcome_message()]

for message in st.session_state.messages:
    avatar = "⚡" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if len(st.session_state.messages) > 1:
    suggestions = FOLLOW_UPS.get(st.session_state.last_intent, FOLLOW_UPS["general"])
    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"fu_{len(st.session_state.messages)}_{i}"):
                st.session_state.pending_question = s
                st.rerun()

user_input = st.chat_input("Ask about EV charging — postcode or place name works...")

question_to_process = None
if user_input:
    question_to_process = user_input
elif st.session_state.pending_question:
    question_to_process = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_process:
    with st.chat_message("user", avatar="👤"):
        st.markdown(question_to_process)
    st.session_state.messages.append({"role": "user", "content": question_to_process})

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Analysing..."):
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

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_current_chat()
    st.rerun()
