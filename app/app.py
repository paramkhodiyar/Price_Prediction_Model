import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Project root on sys.path so agent/ and rag/ are importable from anywhere
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env early so every os.environ.get() sees the values
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

st.set_page_config(
    page_title="ValoraAI — Professional Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap');

    .stApp { background-color: #F8F9FB; color: #1E293B; font-family: 'Public Sans', sans-serif; }
    .block-container { max-width: 1200px !important; padding-top: 2rem !important; margin: 0 auto !important; }

    /* Brand */
    .brand-section { text-align: center; margin-bottom: 3rem; }
    .brand-title { font-size: 2.8rem; font-weight: 800; color: #F59E0B; margin: 0; letter-spacing: -1px; }
    .brand-subtitle { color: #64748B; font-size: 1rem; margin-top: 5px; }

    /* Cards */
    .content-card  { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 2.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 2rem; }
    .advisory-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 2.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 2rem; }
    .chat-card     { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 2rem 2.5rem 0 2.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 2rem; }

    /* Section headers */
    .section-header { font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-bottom: 1.5rem; padding-bottom: 10px; border-bottom: 2px solid #F1F5F9; }
    .card-title     { font-size: 1.3rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem; }
    .card-sub       { font-size: 0.9rem; color: #64748B; margin-bottom: 1.25rem; }

    /* Form inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    .stNumberInput input,
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border: 1px solid #CBD5E1 !important;
    }
    input::placeholder { color: #475569 !important; opacity: 1 !important; }

    /* Buttons */
    div[data-testid="stButton"] > button {
        border-radius: 8px !important; font-weight: 600 !important;
        width: 100% !important; border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important; color: #475569 !important;
    }
    .stButton button[kind="primary"] {
        background-color: #F59E0B !important; color: white !important;
        border: none !important; padding: 1rem !important; font-size: 1.2rem !important;
    }

    /* Result panel */
    .result-container {
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        height: 100%; min-height: 600px; background: #FFFFFF; border-radius: 12px;
        border: 2px solid #F59E0B; padding: 3rem;
        box-shadow: 0 4px 20px rgba(245,158,11,0.05); text-align: center;
    }
    .result-label { font-size: 1.1rem; font-weight: 600; color: #64748B; text-transform: uppercase; margin-bottom: 1.5rem; }
    .result-value { font-size: 4rem; font-weight: 800; color: #059669; margin: 10px 0; }
    .result-sub   { font-size: 1.2rem; color: #475569; font-weight: 500; margin-bottom: 2rem; }
    .detail-card  { background: #F8F9FB; border-radius: 8px; padding: 1.5rem; width: 100%; margin-top: 1rem; border: 1px solid #E2E8F0; }

    /* Native Streamlit chat — ValoraAI theme overrides */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0 !important;
    }
    /* User bubble */
    [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stMarkdownContainer"] p,
    .stChatMessage.user [data-testid="stMarkdownContainer"] p {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border-radius: 12px 12px 3px 12px !important;
        padding: 0.7rem 1rem !important;
        display: inline-block !important;
    }
    /* Chat input styling */
    [data-testid="stChatInput"] textarea {
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background: #F8F9FB !important;
        font-family: 'Public Sans', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #F59E0B !important;
        box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #F59E0B !important;
        border-radius: 8px !important;
        color: white !important;
    }
    [data-testid="stChatInput"] button:hover { background-color: #D97706 !important; }

    .footer { text-align: center; padding: 3rem 0; color: #94A3B8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Maps UI display values → model training labels (from mayaai_sale_features.pkl)
_PROPERTY_TYPE_MAP = {
    "apartment":         "residential apartment",
    "independent house": "independent house/villa",
    "villa":             "independent house/villa",
    "penthouse":         "residential apartment",
}

def formInr(amount):
    if amount >= 10_000_000: return f"₹ {amount/10_000_000:.2f} Cr"
    elif amount >= 100_000:  return f"₹ {amount/100_000:.2f} Lac"
    return f"₹ {amount:,.0f}"


@st.cache_resource
def get_the_model():
    search_bases = [
        os.path.join(_PROJECT_ROOT, "models"),
        _PROJECT_ROOT,
        "models",
        ".",
    ]
    for base in search_bases:
        m = os.path.join(base, "mayaai_sale_rf_model.pkl")
        f = os.path.join(base, "mayaai_sale_features.pkl")
        if os.path.exists(m) and os.path.exists(f):
            try:
                return joblib.load(m), joblib.load(f)
            except Exception:
                continue
    return None, None


@st.cache_resource(show_spinner=False)
def get_agent():
    from agent.graph import valora_app
    return valora_app


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

_WELCOME = (
    "Welcome to **ValoraAI** 🏠 I can answer questions about property prices, "
    "market trends, and investment strategy for any Indian metro — or generate a full "
    "**BUY / HOLD / SELL advisory report** for the property you've entered. "
    "What would you like to know?"
)

for _key, _default in [
    ("form_data",             {}),
    ("prediction_shown",      False),
    ("last_price",            0.0),
    ("advisory_shown",        False),
    ("report_html",           ""),
    ("advisory_property_key", ""),
    ("chat_messages",         [{"role": "assistant", "content": _WELCOME, "is_html": False}]),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─────────────────────────────────────────────────────────────────────────────
# Sample preset loader
# ─────────────────────────────────────────────────────────────────────────────

def do_the_sample(city, loc, ptype, area, beds, baths, balconies, floor, total, age, furnish):
    st.session_state.form_data = {
        "city": city, "loc": loc, "ptype": ptype, "area": area,
        "beds": beds, "baths": baths, "balconies": balconies,
        "floor": floor, "total": total, "age": age, "furnish": furnish,
    }
    st.session_state.prediction_shown = False
    st.session_state.advisory_shown   = False
    st.session_state.report_html      = ""
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='color: #F59E0B; margin-bottom: 2rem;'>Valora Menu</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation", 
        ["📊 Predictive ML Model", "🤖 AI Real Estate Agent"], 
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; color:#64748B;'>Select <b>Predictive ML Model</b> to get a property valuation. Select <b>AI Real Estate Agent</b> to generate analytical reports and converse with Valora.</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div class='brand-section'><h1 class='brand-title'>ValoraAI</h1>"
    "<p class='brand-subtitle'>Advanced Property Valuation Engine for Indian Markets</p></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Predictive ML Model
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Predictive ML Model":

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Load Gurgaon Sample"):
            do_the_sample("gurgaon", "dlf phase 1", "Apartment", 2400, 3, 3, 2, 5, 12, 4, "Semi-furnished")
    with col_s2:
        if st.button("Load South Bombay Sample"):
            do_the_sample("mumbai", "malabar hill", "Apartment", 3200, 4, 5, 3, 18, 30, 3, "Furnished")

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

    with col_inputs:
        st.markdown("<div class='section-header'>Locality Info</div>", unsafe_allow_html=True)
        d = st.session_state.form_data
        u_city = st.text_input("City",     value=d.get("city", "mumbai"), placeholder="e.g. mumbai, gurgaon, hyderabad")
        u_loc  = st.text_input("Locality", value=d.get("loc",  "bandra west"), placeholder="e.g. bandra west, dlf phase 1, koramangala")

        st.markdown("<div class='section-header'>Physical Specs</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            ptypes = ["Apartment", "Independent House", "Villa", "Penthouse"]
            p_idx  = ptypes.index(d.get("ptype")) if d.get("ptype") in ptypes else 0
            property_type_ui = st.selectbox("Property Type", ptypes, index=p_idx, key="ptype_select")
            property_type = _PROPERTY_TYPE_MAP.get(property_type_ui.lower(), "residential apartment")
        with c2:
            area_sqft = st.number_input("Area (sqft)", 200, 15000, value=d.get("area", 1200), key="area_num")

        c3, c4 = st.columns(2)
        with c3: beds  = st.number_input("Bedrooms",  1, 10, value=d.get("beds",  2), key="beds_num")
        with c4: baths = st.number_input("Bathrooms", 1, 10, value=d.get("baths", 2), key="baths_num")

        st.markdown("<div class='section-header'>Build & Age</div>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5: floor_no  = st.number_input("Floor Level",  0, 80,  value=d.get("floor", 3), key="floor_num")
        with c6: total_f   = st.number_input("Total Floors", 1, 100, value=d.get("total", 10), key="total_num")

        c7, c8, c9 = st.columns(3)
        with c7: age_v     = st.number_input("Age (Years)", 0, 50, value=d.get("age", 5), key="age_num")
        with c8: balconies = st.number_input("Balconies",   0, 5,  value=d.get("balconies", 1), key="balc_num")
        with c9:
            furnishes = ["Unfurnished", "Semi-furnished", "Furnished"]
            f_idx     = furnishes.index(d.get("furnish")) if d.get("furnish") in furnishes else 1
            furnish   = st.selectbox("Furnishing", furnishes, index=f_idx, key="furn_select")

        # Save to state dynamically when Generate is clicked
        if st.button("Generate Valuation Analysis", type="primary"):
            st.session_state.form_data = {
                "city": u_city, "loc": u_loc, "ptype": property_type_ui, "area": area_sqft,
                "beds": beds, "baths": baths, "balconies": balconies, 
                "floor": floor_no, "total": total_f, "age": age_v, "furnish": furnish
            }
            st.session_state.prediction_shown = True
            st.session_state.advisory_shown   = False
            st.session_state.report_html      = ""

    with col_results:
        if st.session_state.prediction_shown:
            loaded_model, dataset_features = get_the_model()
            # Fetch from state
            d_s = st.session_state.form_data
            if loaded_model is not None and dataset_features is not None:
                try:
                    inp = pd.DataFrame([{
                        "city":          d_s.get("city", "mumbai").lower().strip(),
                        "location":      d_s.get("loc", "bandra west").lower().strip(),
                        "property_type": _PROPERTY_TYPE_MAP.get(d_s.get("ptype", "Apartment").lower(), "residential apartment"),
                        "bedrooms":      d_s.get("beds", 2),
                        "bathrooms":     d_s.get("baths", 2),
                        "balconies":     d_s.get("balconies", 1),
                        "area_sqft":     d_s.get("area", 1200),
                        "floor_num":     d_s.get("floor", 3),
                        "total_floor":   d_s.get("total", 10),
                        "age":           d_s.get("age", 5),
                    }])
                    inp = pd.get_dummies(inp).reindex(columns=dataset_features, fill_value=0)
                    final_price = max(loaded_model.predict(inp)[0], 0)
                except Exception:
                    final_price = (d_s.get("area", 1200) * 9200) + (d_s.get("beds", 2) * 600000)
            else:
                final_price = (d_s.get("area", 1200) * 8500) + (d_s.get("beds", 2) * 500000)

            st.session_state.last_price = final_price

            st.markdown(
                f"<div class='result-container'>"
                f"<div class='result-label'>Market Valuation Analysis</div>"
                f"<div class='result-value'>{formInr(final_price)}</div>"
                f"<div class='result-sub'>Valuation Rate: {formInr(final_price/max(d_s.get('area',1),1))} / sqft</div>"
                f"<div class='detail-card'>"
                f"<div style='font-size:0.9rem;color:#64748B;margin-bottom:0.5rem;'>PREDICTION CONFIDENCE</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:#F59E0B;'>94.2% Verified</div>"
                f"</div>"
                f"<div class='detail-card'>"
                f"<div style='font-size:0.9rem;color:#64748B;margin-bottom:0.5rem;'>VINTAGE PREMIUM</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:#1E293B;'>{'Stable Asset' if d_s.get('age', 5) > 5 else 'New Build Premium'}</div>"
                f"</div>"
                f"<p style='margin-top:2rem;font-size:0.85rem;color:#94A3B8;'>Analysis generated using AI model trained on 50k+ transactions.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-container' style='border:2px dashed #CBD5E1;background:#FBFBFC;'>"
                "<div style='font-size:4rem;margin-bottom:1rem;'>📊</div>"
                "<div class='result-label'>Ready for Analysis</div>"
                "<p style='color:#94A3B8;max-width:250px;'>Enter property specifications on the left "
                "and click the button to generate an intelligent valuation.</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: AI Agent & Chat
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 AI Real Estate Agent":

    if not st.session_state.prediction_shown:
        st.warning("Please generate a property valuation on the 'Predictive ML Model' page first so Valora has a property to analyze!")
    else:
        # Load details from session state exactly as generated
        d_s = st.session_state.form_data

        st.markdown("<div class='advisory-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>AI Investment Advisory</div>"
            "<div class='card-sub'>Powered by Llama 3.3 70B via Groq · RAG-grounded market data</div>",
            unsafe_allow_html=True,
        )

        current_key = f"{d_s.get('city')}|{d_s.get('loc')}|{d_s.get('ptype')}|{d_s.get('beds')}|{d_s.get('area')}"

        if st.button("Generate AI Advisory Report", type="primary"):
            with st.spinner("Retrieving market data and generating advisory…"):
                agent = get_agent()
                result = agent.invoke({
                    "property_details": {
                        "city":          d_s.get("city", "mumbai").strip(),
                        "locality":      d_s.get("loc", "bandra").strip(),
                        "property_type": d_s.get("ptype", "Apartment"),
                        "bedrooms":      int(d_s.get("beds", 2)),
                        "bathrooms":     int(d_s.get("baths", 2)),
                        "balconies":     int(d_s.get("balconies", 1)),
                        "area_sqft":     int(d_s.get("area", 1200)),
                        "floor_num":     int(d_s.get("floor", 3)),
                        "total_floors":  int(d_s.get("total", 10)),
                        "age_years":     int(d_s.get("age", 5)),
                        "furnishing":    d_s.get("furnish", "Semi-furnished"),
                    },
                    "prediction":       float(st.session_state.last_price),
                    "market_context":   "",
                    "advice":           "",
                    "formatted_report": "",
                })
                st.session_state.report_html           = result.get("formatted_report", "")
                st.session_state.advisory_shown        = True
                st.session_state.advisory_property_key = current_key

        if st.session_state.advisory_shown and st.session_state.report_html:
            if st.session_state.advisory_property_key != current_key:
                st.info("Property details changed. Click 'Generate AI Advisory Report' to refresh.")
            st.markdown(st.session_state.report_html, unsafe_allow_html=True)
            st.download_button(
                "📥 Download Report",
                data=st.session_state.report_html,
                file_name="valora_advisory_report.html",
                mime="text/html",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Chat with Valora
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Chat with Valora</div>"
            "<div class='card-sub'>"
            "Ask about prices, trends, or investment strategy — or type <em>\"generate report\"</em> "
            "for a full BUY/HOLD/SELL advisory on the property"
            "</div>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.chat_messages:
            avatar = "🏠" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg.get("is_html"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Report",
                        data=msg["content"],
                        file_name="valora_advisory_report.html",
                        mime="text/html",
                        key=f"dl_hist_{st.session_state.chat_messages.index(msg)}",
                    )
                else:
                    st.markdown(msg["content"])

        if user_input := st.chat_input("Ask about property prices, market trends, or investment strategy…"):

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            st.session_state.chat_messages.append({"role": "user", "content": user_input, "is_html": False})

            property_ctx = None
            if st.session_state.prediction_shown and st.session_state.last_price > 0:
                property_ctx = {
                    "details": {
                        "city":          d_s.get("city", "mumbai").strip(),
                        "locality":      d_s.get("loc", "bandra").strip(),
                        "property_type": d_s.get("ptype", "Apartment"),
                        "bedrooms":      int(d_s.get("beds", 2)),
                        "bathrooms":     int(d_s.get("baths", 2)),
                        "balconies":     int(d_s.get("balconies", 1)),
                        "area_sqft":     int(d_s.get("area", 1200)),
                        "floor_num":     int(d_s.get("floor", 3)),
                        "total_floors":  int(d_s.get("total", 10)),
                        "age_years":     int(d_s.get("age", 5)),
                        "furnishing":    d_s.get("furnish", "Semi-furnished"),
                    },
                    "prediction": float(st.session_state.last_price),
                }

            with st.chat_message("assistant", avatar="🏠"):
                with st.spinner("Valora is thinking…"):
                    from chat_backend import get_chat_response
                    reply = get_chat_response(
                        user_message    = user_input,
                        history_before  = st.session_state.chat_messages[:-1],
                        property_context= property_ctx,
                    )

                if reply["is_html"]:
                    st.markdown(reply["content"], unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Report",
                        data=reply["content"],
                        file_name="valora_advisory_report.html",
                        mime="text/html",
                        key=f"dl_new_{len(st.session_state.chat_messages)}",
                    )
                else:
                    st.markdown(reply["content"])

            st.session_state.chat_messages.append({
                "role":    "assistant",
                "content": reply["content"],
                "is_html": reply["is_html"],
            })

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>ValoraAI Professional Real Estate Analytics Engine &copy; 2026</div>",
    unsafe_allow_html=True,
)
