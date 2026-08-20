import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
load_dotenv()
# Agent State
class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]
# Tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
tools = [add]

# Model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)
# Model Call
def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content=(
            "You are my AI assistant. "
            "Please answer my query to the best of your ability."
        )
    )
    response = model.invoke(
        [system_message] + list(state["messages"])
    )
    return {
        "messages": [response]
    }
# Decide What To Do Next
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"
# Create Graph
graph = StateGraph(AgentState)
# Agent node
graph.add_node(
    "our_agent",
    model_call
)
# Tool node
tool_node = ToolNode(tools=tools)

graph.add_node(
    "tools",
    tool_node
)
# Start
graph.set_entry_point("our_agent")
# Conditional edge
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
# After tool → Agent again
graph.add_edge(
    "tools",
    "our_agent"
)
# Compile
app = graph.compile()
# Print Stream
def print_stream(stream):

    for state in stream:

        messages = state["messages"]

        last_message = messages[-1]

        last_message.pretty_print()
# Input
inputs = {
    "messages": [
        (
            "user",
            "Add 40 + 12 and then tell me a joke please and then add 3+5"
        )
    ]
}
# Run
print_stream(
    app.stream(
        inputs,
        stream_mode="values"
    )
)