from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

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
