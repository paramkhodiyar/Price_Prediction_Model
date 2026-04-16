from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agent.llm import llm
from langchain_core.messages import SystemMessage, HumanMessage

class AgentState(TypedDict):
    property_details: dict
    prediction: float
    advice: str
    report: str

def analyze_property(state: AgentState) -> AgentState:
    return state

def fetch_prediction(state: AgentState) -> AgentState:
    import os
    import pickle
    details = state.get("property_details", {})
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mayaai_sale_rf_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        try:
            pred = model.predict([list(details.values())])
        except:
            pred = model.predict([details])
            
        if pred is not None and len(pred) > 0:
            state["prediction"] = float(pred[0])
    except Exception:
        # Silent fallback to current state prediction if model fails or isn't built yet
        pass
    return state

def generate_advice(state: AgentState) -> AgentState:
    details = state.get("property_details", {})
    prediction = state.get("prediction", 0)

    prompt = f"""
Analyze the following Indian real estate investment opportunity.

INPUT DATA:
Property Details:
{details}

Predicted Market Value:
₹{prediction:,.2f}

TASK:
Evaluate this property strictly as an Indian real estate investment advisor.

DECISION RULES:
- BUY → If property shows strong appreciation/investment potential, favorable market trend, and manageable risk.
- HOLD → If property appears stable/moderately valuable but lacks strong upside.
- SELL → If property appears overvalued, risky, depreciating, or poor investment.

RISK RULES:
- Low → Strong fundamentals, low downside, positive outlook.
- Medium → Balanced opportunity with manageable uncertainty.
- High → Significant uncertainty, downside, depreciation, or investment concern.

ANALYSIS FACTORS:
Consider ALL of the following:
1. Location desirability
2. Property type attractiveness
3. Size/value proposition
4. Physical condition
5. Market trend outlook
6. Risk/reward potential

OUTPUT REQUIREMENTS:
- Respond ONLY with raw valid JSON.
- No markdown.
- No extra commentary.
- No text outside JSON.
- Ensure all fields are present.
- Keep reasoning concise but professional.

REQUIRED JSON FORMAT:
{{
  "verdict": "BUY/HOLD/SELL",
  "reason": "2-3 sentence professional investment reasoning",
  "risk": "LOW/MEDIUM/HIGH",
  "summary": "1 sentence investor recommendation",
  "disclaimer": "This is AI-generated analysis and not financial advice."
}}
"""

    messages = [
        SystemMessage(
            content="""
You are Valora, an elite Indian real estate investment analysis engine.

RULES:
- Always return STRICT valid JSON only.
- Never wrap response in markdown/code blocks.
- Never include explanation outside JSON.
- Follow decision rules exactly.
- Base reasoning only on provided input.
- Maintain institutional/professional tone.
"""
        ),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    state["advice"] = response.content

    return state
def format_report(state: AgentState) -> AgentState:
    return state

workflow = StateGraph(AgentState)

workflow.add_node("analyze_property", analyze_property)
workflow.add_node("fetch_prediction", fetch_prediction)
workflow.add_node("generate_advice", generate_advice)
workflow.add_node("format_report", format_report)

workflow.add_edge(START, "analyze_property")
workflow.add_edge("analyze_property", "fetch_prediction")
workflow.add_edge("fetch_prediction", "generate_advice")
workflow.add_edge("generate_advice", "format_report")
workflow.add_edge("format_report", END)

valora_app = workflow.compile()

# if __name__ == "__main__":
#     import json

#     test_cases = [
#         {
#             "property_details": {
#                 "location": "Mumbai",
#                 "type": "2BHK",
#                 "size_sqft": 800,
#                 "condition": "good",
#                 "current_market_trend": "appreciation"
#             },
#             "prediction": 15000000.0,
#             "advice": "",
#             "report": ""
#         },
#         {
#             "property_details": {
#                 "location": "Delhi",
#                 "type": "3BHK",
#                 "size_sqft": 1200,
#                 "condition": "excellent",
#                 "current_market_trend": "stable"
#             },
#             "prediction": 22000000.0,
#             "advice": "",
#             "report": ""
#         },
#         {
#             "property_details": {
#                 "location": "Bangalore",
#                 "type": "Villa",
#                 "size_sqft": 2500,
#                 "condition": "luxury",
#                 "current_market_trend": "high appreciation"
#             },
#             "prediction": 45000000.0,
#             "advice": "",
#             "report": ""
#         }
#     ]

#     for idx, test_case in enumerate(test_cases, start=1):
#         print(f"\n{'='*50}")
#         print(f"Running Test Case {idx}: {test_case['property_details']['location']}")
#         print(f"{'='*50}")

#         result = valora_app.invoke(test_case)

#         # Force parse json and prettify
#         raw_advice = result.get("advice", "").strip()
#         if raw_advice.startswith("```json"):
#             raw_advice = raw_advice[7:-3].strip()
#         elif raw_advice.startswith("```"):
#             raw_advice = raw_advice[3:-3].strip()

#         try:
#             parsed = json.loads(raw_advice)
#             print("\nGenerated Advice (Valid JSON):\n")
#             print(json.dumps(parsed, indent=2))
#         except json.JSONDecodeError:
#             print("\nGenerated Advice (Failed to parse JSON):\n")
#             print(raw_advice)


