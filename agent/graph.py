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
    return state

def generate_advice(state: AgentState) -> AgentState:
    details = state.get("property_details", {})
    prediction = state.get("prediction", 0)

    prompt = f"""
You are evaluating a real estate investment opportunity.

PROPERTY DATA:
{details}

ESTIMATED MARKET VALUE:
₹{prediction:,.2f}

TASK:
Analyze this property as a senior real estate investment advisor.

You must evaluate based on:
1. Property condition and physical quality
2. Location attractiveness and demand potential
3. Current market trend implications
4. Estimated value vs investment attractiveness
5. Risk factors for buyer/investor

INSTRUCTIONS:
- Determine whether the recommendation is BUY, HOLD, or SELL.
- Justify your recommendation with concise financial and strategic reasoning.
- Mention realistic market/investment/legal considerations where relevant.
- Be practical and analytical, not generic.
- Avoid speculation beyond provided data.
- Do NOT mention lack of information.
- Assume all provided data is accurate.

OUTPUT FORMAT STRICTLY:

VERDICT: [BUY/HOLD/SELL]

REASONING:
1. [Financial / valuation reasoning]
2. [Market / location reasoning]
3. [Risk / strategic reasoning]

SUMMARY:
[One sentence final investor recommendation.]
"""

    messages = [
        SystemMessage(
            content="""
You are Valora, an elite real estate valuation and investment strategist with expertise in:
- Property investment analysis
- Financial risk assessment
- Market trend forecasting
- Real estate portfolio strategy

Your responses must be:
- Highly professional
- Data-driven
- Concise but insightful
- Structured exactly as requested
- Free from hallucinations or unnecessary filler.
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
#         },

#         {
#             "property_details": {
#                 "location": "Pune",
#                 "type": "1BHK",
#                 "size_sqft": 500,
#                 "condition": "average",
#                 "current_market_trend": "depreciation"
#             },
#             "prediction": 6000000.0,
#             "advice": "",
#             "report": ""
#         }
#     ]


#     for idx, test_case in enumerate(test_cases, start=1):
#         print(f"\n{'='*50}")
#         print(f"Running Test Case {idx}")
#         print(f"{'='*50}")

#         result = valora_app.invoke(test_case)

#         print("\nGenerated Advice:\n")
#         print(result.get("advice", "No advice generated."))

#         print("\nGenerated Report:\n")
#         print(result.get("report", "No report generated."))