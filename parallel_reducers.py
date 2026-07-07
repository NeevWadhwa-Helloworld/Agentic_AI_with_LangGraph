import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}


class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]


def toxicity_node(state: AnalyzerState) -> dict:
    print("\n--- Executing Toxicity Node ---")
    prompt = (
        "You are a content-safety specialist focused on toxicity detection. "
        "For the given text, output a single integer from 0 to 100 representing the toxicity level (0 = no toxicity, 100 = extremely toxic). "
        "Do NOT include any explanation, labels, or extra text — return ONLY the integer number.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score": {"toxicity_level": score}}


def copyright_node(state: AnalyzerState) -> dict:
    print("\n--- Executing Copyright Node ---")
    prompt = (
        "You are a content-safety specialist focused on copyright and plagiarism risk. "
        "For the given text, output a single integer from 0 to 100 representing the copyright/plagiarism risk (0 = no risk, 100 = very high risk). "
        "Do NOT include any explanation, labels, or extra text — return ONLY the integer number.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score": {"copyright_level": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n--- Executing Culture Node ---")
    prompt = (
        "You are a cultural-sensitivity specialist. For the given text, output a single integer from 0 to 100 representing cultural insensitivity/offense (0 = not insensitive, 100 = highly offensive). "
        "Do NOT include any explanation, labels, or extra text — return ONLY the integer number.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score": {"culture_level": score}}


builder = StateGraph(AnalyzerState)

builder.add_node("toxicity", toxicity_node)
builder.add_node("copyright", copyright_node)
builder.add_node("culture", culture_node)

builder.add_edge(START, "toxicity")
builder.add_edge(START, "copyright")
builder.add_edge(START, "culture")

builder.add_edge("toxicity", END)
builder.add_edge("copyright", END)
builder.add_edge("culture", END)

app = builder.compile()

sample_script = "Yo guys! I hate you all. This is the worst video ever. I can't believe you made me watch this garbage. You're all idiots."

initial_state = {"raw_text": sample_script, "safety_score": {}}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])
