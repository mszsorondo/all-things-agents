from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.types import Command
import os
import json
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

# Get keys for your project from the project settings page: https://cloud.langfuse.com

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-62b9334a-089d-433d-8722-c4582e632796"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-2223a7f4-75d0-4125-b8e7-6843940a54bb"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
# Your openai key



from langfuse import get_client
 
langfuse = get_client()

from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

#based on https://langchain-ai.github.io/langgraph/concepts/multi_agent/

model = ChatOpenAI(model="gpt-4o-mini", api_key="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")

def coding_agent(state: MessagesState):
    response = model.invoke([SystemMessage(
        content="You are a coding agent. \
            You are given a task and you need to code the solution in python."), 
            state["messages"][-1]])
    goto = "testing_agent"
    return Command(goto=goto, update={"messages": [response]})

def testing_agent(state: MessagesState):
    response = model.invoke([SystemMessage(
        content="You are a testing agent. \
            You are given a code and you need to test the solution."), 
            state["messages"][-1]])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(coding_agent)
builder.add_node(testing_agent, state_schema=MessagesState)
# define the flow explicitly
builder.add_edge(START, "coding_agent")
builder.add_edge("coding_agent", "testing_agent")

graph = builder.compile()

# Run the graph and capture the result
result = graph.invoke(
    {"messages": [HumanMessage(content="Make a calculator capable of: adding, subtracting, multiplying, and dividing.")]},
    config={"callbacks": [langfuse_handler]},
)

# Get the final state
print("Final state:", result)

#store result in a JSON file
with open("result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)