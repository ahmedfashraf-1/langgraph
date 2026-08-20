from typing import TypedDict,List
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
import os
# from groq import Groq
load_dotenv()
#
# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )
# models = client.models.list()
# for model in models.data:
#     print(model.id)
class AgentState(TypedDict):
    messages: List[BaseMessage]
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
def process_message(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAgent Message: {response.content}")
    return state
graph=StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()
user_input=input("Enter your message: ")
result=agent.invoke({"messages":[HumanMessage(content=user_input)]})
