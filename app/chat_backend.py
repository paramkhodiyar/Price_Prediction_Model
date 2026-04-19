"""
Chat backend — runs entirely inside the Streamlit process.
No external endpoint, no CORS, no iframe.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Keywords that trigger a full advisory report via the LangGraph agent
_REPORT_TRIGGERS = {
    "advisory", "report", "analyse", "analyze", "verdict",
    "should i buy", "should i sell", "should i invest",
    "buy or sell", "generate report", "full report",
    "investment report", "property report",
}


def _wants_report(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _REPORT_TRIGGERS)


def _city_hint(user_message: str, property_context: dict) -> str:
    """Best-effort city extraction for RAG query."""
    if property_context:
        city = property_context.get("details", {}).get("city", "")
        if city:
            return city
    # Capitalised word heuristic
    for word in user_message.split():
        if len(word) > 2 and word[0].isupper():
            return word
    return ""


def generate_advisory_report(property_context: dict) -> dict:
    """
    Run the full LangGraph agent and return the formatted HTML report.
    property_context = {"details": {...}, "prediction": float}
    """
    from agent.graph import valora_app

    details    = property_context.get("details", {})
    prediction = property_context.get("prediction", 0.0)

    result = valora_app.invoke({
        "property_details": details,
        "prediction":       prediction,
        "market_context":   "",
        "advice":           "",
        "formatted_report": "",
    })

    html = result.get("formatted_report", "")
    if html:
        return {"content": html, "is_html": True}

    return {"content": "Could not generate report. Please try again.", "is_html": False}


def get_chat_response(user_message: str, history_before: list, property_context: dict = None) -> dict:
    """
    Conversational response via Groq/Llama with RAG context.

    Parameters
    ----------
    user_message     : the new user message (NOT yet in history_before)
    history_before   : list of {"role", "content", "is_html"} BEFORE current message
    property_context : {"details": dict, "prediction": float} or None

    Returns {"content": str, "is_html": bool}
    """
    # Full report requested and we have a valued property → run agent
    if _wants_report(user_message) and property_context and property_context.get("prediction", 0) > 0:
        return generate_advisory_report(property_context)

    from agent.llm import llm
    from rag.retriever import get_market_context

    # Build RAG query
    details    = property_context.get("details", {}) if property_context else {}
    city       = _city_hint(user_message, property_context)
    prop_type  = details.get("property_type", "apartment")
    bedrooms   = int(details.get("bedrooms", 2))
    rag_ctx    = get_market_context(city or user_message[:60], prop_type, bedrooms)

    # Property snapshot for system prompt
    prop_block = ""
    if property_context and property_context.get("prediction", 0) > 0:
        pred = property_context["prediction"]
        prop_block = (
            f"\n\nProperty currently under analysis:\n"
            f"{details}\n"
            f"Predicted market value: ₹{pred:,.0f}"
        )

    system_content = f"""You are Valora, the expert AI assistant of ValoraAI — India's leading property intelligence platform.

Your expertise:
- Indian real estate prices, micro-market trends, and investment analysis
- All property types: apartments, villas, independent houses, penthouses
- Key metros: Mumbai, Gurgaon, Delhi NCR, Hyderabad, Bangalore, Pune, Kolkata, Chennai
- Investment concepts: rental yield, capital appreciation, RERA, stamp duty, home loans
- Practical, opinionated advice tailored to Indian market conditions

Comparable market data (RAG-retrieved for this query):
{rag_ctx}
{prop_block}

Rules:
- Be concise, direct, and data-driven. No waffle.
- Use ₹ Cr / Lac notation.
- Give a clear BUY / HOLD / SELL view when asked about investment potential.
- If the user explicitly asks for a "report" or "advisory", tell them to use the
  "Generate AI Advisory Report" button above the chat for the full formatted report.
- Never refuse a real estate question.
"""

    messages = [SystemMessage(content=system_content)]
    for msg in history_before[-8:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and not msg.get("is_html"):
            messages.append(AIMessage(content=str(msg["content"])[:600]))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return {"content": response.content, "is_html": False}
