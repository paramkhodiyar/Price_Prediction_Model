from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agent.llm import llm
from langchain_core.messages import SystemMessage, HumanMessage


class AgentState(TypedDict):
    property_details: dict
    prediction: float
    market_context: str       # injected from RAG retriever
    advice: str               # raw JSON from LLM
    formatted_report: str     # final HTML report


def analyze_property(state: AgentState) -> AgentState:
    return state


def fetch_prediction(state: AgentState) -> AgentState:
    """Use prediction already supplied by the app; fall back to RF model if missing."""
    if state.get("prediction", 0) > 0:
        return state  # prediction passed in from app.py — skip re-inference

    import os, pickle
    details = state.get("property_details", {})
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mayaai_sale_rf_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        try:
            pred = model.predict([list(details.values())])
        except Exception:
            pred = model.predict([details])
        if pred is not None and len(pred) > 0:
            state["prediction"] = float(pred[0])
    except Exception:
        pass
    return state


def retrieve_market_context(state: AgentState) -> AgentState:
    """Fetch top-3 RAG results and store in state for the LLM prompt."""
    from rag.retriever import get_market_context

    details = state.get("property_details", {})
    city          = details.get("city", "")
    property_type = details.get("property_type", "apartment")
    bedrooms      = int(details.get("bedrooms", 2))

    state["market_context"] = get_market_context(city, property_type, bedrooms)
    return state


def generate_advice(state: AgentState) -> AgentState:
    details        = state.get("property_details", {})
    prediction     = state.get("prediction", 0)
    market_context = state.get("market_context", "")

    prompt = f"""
Analyze the following Indian real estate investment opportunity.

INPUT DATA:
Property Details:
{details}

Predicted Market Value:
₹{prediction:,.2f}

COMPARABLE MARKET DATA (from RAG retrieval):
{market_context}

TASK:
Evaluate this property strictly as an Indian real estate investment advisor.
Use the comparable market data above to ground your analysis in real market conditions.

DECISION RULES:
- BUY  → Strong appreciation/investment potential, favorable market trend, manageable risk.
- HOLD → Stable/moderately valuable but lacks strong upside or has moderate risk.
- SELL → Overvalued, risky, depreciating, or poor investment.

RISK RULES:
- LOW    → Strong fundamentals, low downside, positive outlook.
- MEDIUM → Balanced opportunity with manageable uncertainty.
- HIGH   → Significant uncertainty, downside, depreciation, or investment concern.

ANALYSIS FACTORS:
1. Location desirability vs comparable market data
2. Property type attractiveness in this micro-market
3. Size/value proposition relative to comparable prices
4. Physical condition (age, furnishing, floor)
5. Market trend outlook for this city/locality
6. Risk/reward potential

OUTPUT REQUIREMENTS:
- Respond ONLY with raw valid JSON. No markdown. No extra text outside JSON.
- Ensure all fields are present and concise.

REQUIRED JSON FORMAT:
{{
  "verdict": "BUY/HOLD/SELL",
  "reason": "2-3 sentence professional investment reasoning referencing the market data",
  "risk": "LOW/MEDIUM/HIGH",
  "summary": "1 sentence investor recommendation",
  "disclaimer": "This is AI-generated analysis and not financial advice."
}}
"""

    messages = [
        SystemMessage(content=(
            "You are Valora, an elite Indian real estate investment analysis engine. "
            "Always return STRICT valid JSON only. Never wrap response in markdown or code blocks. "
            "Never include explanation outside JSON. Maintain institutional/professional tone."
        )),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    state["advice"] = response.content
    return state


def format_report(state: AgentState) -> AgentState:
    """Render the LLM JSON advice into a styled HTML report."""
    from rag.report import format_report as render_html

    advice         = state.get("advice", "{}")
    market_context = state.get("market_context", "")

    # Inject market_context into the report for display
    import json
    try:
        raw = advice.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1][4:].strip() if parts[1].startswith("json") else parts[1].strip()
        data = json.loads(raw)
    except Exception:
        data = {"verdict": "HOLD", "reason": advice, "risk": "MEDIUM",
                "summary": "Analysis complete.", "disclaimer": "AI-generated, not financial advice."}

    data["market_context"] = market_context
    state["formatted_report"] = render_html(json.dumps(data))
    return state


def validate_output(state: AgentState):
    """Conditional edge: check if the advice is valid JSON and has required fields."""
    import json
    try:
        raw = state.get("advice", "").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1][4:].strip() if parts[1].startswith("json") else parts[1].strip()
        data = json.loads(raw)
        if all(k in data for k in ["verdict", "reason", "risk"]):
            return "format_report"
    except Exception:
        pass
    return "generate_advice"  # Retry if invalid


workflow = StateGraph(AgentState)

workflow.add_node("analyze_property",        analyze_property)
workflow.add_node("fetch_prediction",        fetch_prediction)
workflow.add_node("retrieve_market_context",   retrieve_market_context)
workflow.add_node("generate_advice",         generate_advice)
workflow.add_node("format_report",           format_report)

workflow.add_edge(START,                      "analyze_property")
workflow.add_edge("analyze_property",         "fetch_prediction")
workflow.add_edge("fetch_prediction",         "retrieve_market_context")
workflow.add_edge("retrieve_market_context",   "generate_advice")

# The "Agentic" part: Conditional routing for autonomous verification
workflow.add_conditional_edges(
    "generate_advice",
    validate_output,
    {
        "format_report": "format_report",
        "generate_advice": "generate_advice"
    }
)

workflow.add_edge("format_report",            END)

valora_app = workflow.compile()
