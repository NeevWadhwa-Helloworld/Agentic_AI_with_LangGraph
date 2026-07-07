from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from typing import TypedDict


class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


def editor_node(state: pipelinestate) -> dict:
    prompt = (
        "You are an expert copyeditor. Clean up the following raw text in Hinglish. "
        "Fix any grammar, spelling, and punctuation errors. Make the text clear, concise, and natural. "
        "While keeping the core message intact, return only the edited text in Hinglish.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: pipelinestate) -> dict:
    print("\n--- Executing Scriptwriter Node ---")
    prompt = (
        "You are an expert scriptwriter. Create a compelling script based on the following edited text in Hinglish. "
        "Ensure the script is engaging, well-structured, and suitable for a video format. "
        "Use a friendly mix of English and Hindi words naturally. Return only the script text in Hinglish.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}


def translator_node(state: pipelinestate) -> dict:
    print("\n--- Executing Translator Node ---")
    prompt = (
        "You are a professional translator. Rewrite the following script in Hinglish. "
        "Use a natural, friendly mix of English and Hindi words while keeping the original meaning and tone. "
        "Return only the rewritten script text in Hinglish.\n\n"
        f"Script Text:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}


graph = StateGraph(pipelinestate)

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

app = graph.compile()
result = app.invoke({
    "raw_input": "AI agents are the future of technology. They can perform tasks autonomously, learn from data, and interact with humans in natural ways. AI agents are being used in various industries, including healthcare, finance, and customer service. They can analyze large datasets, provide insights, and make decisions faster than humans. As AI technology continues to advance, AI agents will become even more capable and integrated into our daily lives."
})

print("your result are: -\n\n")
print(result['final_output'])
