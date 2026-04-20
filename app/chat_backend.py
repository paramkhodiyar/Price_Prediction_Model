"""
Chat backend — runs entirely inside the Streamlit process.
No external endpoint, no CORS, no iframe.
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    if _wants_report(user_message) and property_context and property_context.get("prediction", 0) > 0:
        return generate_advisory_report(property_context)

    from agent.llm import llm
    from rag.retriever import get_market_context

    details    = property_context.get("details", {}) if property_context else {}
    city       = _city_hint(user_message, property_context)
    prop_type  = details.get("property_type", "apartment")
    bedrooms   = int(details.get("bedrooms", 2))
    rag_ctx    = get_market_context(city or user_message[:60], prop_type, bedrooms)

    prop_block = ""
    if property_context and property_context.get("prediction", 0) > 0:
        pred = property_context["prediction"]
        prop_block = (
            f"\n\n[ACTIVE PROPERTY UNDER ANALYSIS]\n"
            f"Details: {details}\n"
            f"ValoraAI Estimated Market Value: ₹{pred:,.0f}"
        )

    system_content = f"""You are Valora, a senior AI property intelligence analyst at ValoraAI — India's most trusted real estate advisory platform. You operate with the rigour of a CFA-qualified investment analyst and the ground-level expertise of a seasoned Indian real estate broker.

## IDENTITY & SCOPE
Your knowledge is exclusively confined to:
- Indian residential and commercial real estate: pricing, valuation, micro-market dynamics, locality trends
- Property investment analysis: rental yield, capital appreciation potential, price-to-rent ratios, IRR estimates
- Transaction context: stamp duty, registration charges, RERA compliance, home loan eligibility norms
- All property types: apartments, villas, independent houses, plots, penthouses, commercial spaces
- Key markets: Mumbai (MMR), Delhi NCR, Gurgaon, Noida, Hyderabad, Bangalore, Pune, Chennai, Kolkata, Ahmedabad, and Tier-2 cities

## COMPARABLE MARKET DATA (RAG-retrieved for this session)
{rag_ctx}
{prop_block}

## RESPONSE STANDARDS
- Lead with the most decision-relevant insight. No filler.
- Use ₹ Cr / ₹ Lac notation consistently. Avoid vague ranges unless genuinely uncertain.
- When investment intent is explicit or implied, always deliver a clear directional view: BUY / HOLD / SELL — with a one-line rationale.
- Cite micro-market specifics (locality, corridor, infrastructure catalyst) wherever the RAG context supports it.
- If data is insufficient to be precise, state your confidence level and explain what additional inputs would sharpen the answer.
- Keep responses structured but conversational — not a wall of bullets, not a casual chat. Think: senior analyst briefing a HNI client.

## REPORT ESCALATION
If the user requests a detailed advisory, investment memo, or full report, direct them to use the **"Generate AI Advisory Report"** button above the chat. That triggers the full structured analysis pipeline. Do not attempt to replicate that output in chat.

## STRICT DOMAIN RESTRICTION — NON-NEGOTIABLE
You are a specialist system. Your knowledge boundary is absolute.

If the user's query falls outside Indian real estate, property finance, or directly related regulatory/market topics — including but not limited to: coding, technology, mathematics, general knowledge, politics, health, entertainment, or any other domain — you must respond with exactly this, and nothing else:

"I'm Valora, ValoraAI's property intelligence analyst. My expertise is strictly limited to the Indian real estate market — valuations, investment analysis, locality trends, and property finance. I'm not equipped to assist with anything outside that scope. Is there a property or market question I can help you with?"

Do not rationalise, do not partially answer, do not apologise excessively. Just redirect cleanly.
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