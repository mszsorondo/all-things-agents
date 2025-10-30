#from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent

import pm4py
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter

import os

os.environ["ANTHROPIC_API_KEY"] = "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"

hiring_log = hiring = pm4py.read_xes("/home/marco/Escritorio/austral/all-things-agents/pm-samples/benchmark/hiring/hiring_log_high.xes")
hiring_bpmn = pm4py.discover_bpmn_inductive(hiring_log)
breakpoint()
hiring_bpmn_file = bpmn_exporter.apply(hiring_bpmn, "discovered_model.bpmn")
with open("discovered_model.bpmn","r") as prompt:
    hiring_bpmn_str = prompt.read()

#TODO MAYBE IMAGES ARE BETTER THAN TEXT, .BPMN FILE CAN BE FULL OF MISLEADING CHARACTERS


hiring_net, hiring_im, hiring_fm = pm4py.discover_petri_net_inductive(hiring_log)
hiring_dfg = pm4py.llm.abstract_dfg(hiring_log)


# TODO remember to add interpretation instructuons for each model type
# TODO add thinking, DEEPAGENTS library 
with open("miner_prompt_gpt.txt","r") as prompt:
    sys_prompt_base = prompt.read()

def write_bpmn_file(content:str, fname :str):
    """
    Writes BPMN content to a specified file in binary mode.

    Args:
        content (str): The BPMN content to write into the file.
        fname (str): The name (or path) of the file where the content will be written.
    """
    with open(fname, "w") as file:
        file.write(content)

def write_comprehensive_textual_description(content:str, fname:str):
    """
    Writes a comprehensive textual description to a specified file in binary mode.

    Args:
        content (str): The textual content to write into the file.
        fname (str): The name (or path) of the file where the content will be written.
    """
    with open(fname, "w") as file:
        file.write(content)

breakpoint()
graph = create_react_agent(
    model="anthropic:claude-opus-4-1",
    tools=[write_bpmn_file, write_comprehensive_textual_description],
    prompt= sys_prompt_base
)




inputs = {"messages": [{"role": "user", "content": f"\nProcess Mining Representations: \
    1. Petri Net from inductive miner: {str(hiring_net)} \n with initial marking {str(hiring_im)}, final marking {str(hiring_im)},\
    2. BPMN extracted from inductive miner: {hiring_bpmn_str}\n \
    3. DFG (Directly-Follows Graph) abstraction: {hiring_dfg}"}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)


