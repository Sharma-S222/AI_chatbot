import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import (metal_price_tool,convert_currency, web_search)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-lite"
)

db_con = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(db_con)

config = {'configurable': {'thread_id':1}}

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools([metal_price_tool, convert_currency, web_search])

def chatbot(state: State):
    messages = [
        SystemMessage(
            content="""Ask follow-up questions when information is missing"""
        )
    ] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(State)
builder.add_node(chatbot)
builder.add_node("tools", ToolNode([metal_price_tool, convert_currency, web_search]))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)

while True:
    msg = input("\nYou: ")
    if msg in {"exit", "goodbye", "bye"}:
        break
    state = graph.invoke({"messages": [{"role":"user","content":msg}]}, config=config)
    print("Gemini: ")
    print("  ",state["messages"][-1].content)
    