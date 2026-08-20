from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
load_dotenv()
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
def process_message(state: AgentState) -> AgentState:

    response = llm.invoke(state["messages"])

    return {
        "messages": state["messages"] + [response]
    }
graph = StateGraph(AgentState)

graph.add_node("process", process_message)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

print("Chatbot started!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    # Add user message
    conversation_history.append(
        HumanMessage(content=user_input)
    )
    # Run Agent
    result = agent.invoke({
        "messages": conversation_history
    })
    # Update conversation history
    conversation_history = result["messages"]
    # Get AI response
    ai_response = conversation_history[-1]
    print(f"AI: {ai_response.content}\n")
with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Your Conversation Log:\n\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(
                f"You: {message.content}\n"
            )
        elif isinstance(message, AIMessage):
            file.write(
                f"AI: {message.content}\n\n"
            )
print("Conversation saved to logging.txt")