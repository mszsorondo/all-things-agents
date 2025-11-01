#from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
import pm4py
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
import os
import shutil
import json
from datetime import datetime

os.environ["ANTHROPIC_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

hiring_log = hiring = pm4py.read_xes("/home/marco/Escritorio/austral/all-things-agents/pm-samples/benchmark/hiring/hiring_log_high.xes")
hiring_bpmn = pm4py.discover_bpmn_inductive(hiring_log)

hiring_bpmn_file = bpmn_exporter.apply(hiring_bpmn, "discovered_model.bpmn")
with open("discovered_model.bpmn","r") as prompt:
    hiring_bpmn_str = prompt.read()

#TODO MAYBE IMAGES ARE BETTER THAN TEXT, .BPMN FILE CAN BE FULL OF MISLEADING CHARACTERS


hiring_net, hiring_im, hiring_fm = pm4py.discover_petri_net_inductive(hiring_log)
hiring_dfg = pm4py.llm.abstract_dfg(hiring_log)


# TODO remember to add interpretation instructuons for each model type
# TODO add thinking, DEEPAGENTS library 
prompt_file = "new_prompt.txt"
with open(prompt_file, "r") as prompt:
    sys_prompt_base = prompt.read()

# Create runs directory and unique run folder
runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(runs_dir, f"run_{run_timestamp}")
os.makedirs(run_folder, exist_ok=True)

# Store the system prompt file used for this run
shutil.copy(prompt_file, run_folder)

def write_bpmn_file(fname :str, content:str):
    """
    Writes BPMN content to a file in the current run's folder.

    Args:
        fname (str): The name of the file (will be saved in the current run's folder).
        content (str): The BPMN content to write into the file.
        
    """
    filepath = os.path.join(run_folder, fname)
    with open(filepath, "w") as file:
        file.write(content)

def write_comprehensive_textual_description(fname:str, content:str):
    """
    Writes a comprehensive textual description to a file in the current run's folder.

    Args:
        fname (str): The name of the file (will be saved in the current run's folder).
        content (str): The textual content to write into the file.

    """
    filepath = os.path.join(run_folder, fname)
    with open(filepath, "w") as file:
        file.write(content)

# Define tools and model for agent
tools = [write_bpmn_file, write_comprehensive_textual_description]
model_name = "anthropic:claude-sonnet-4-5"  # "openai:gpt-4o-mini"

graph = create_react_agent(
    model=model_name,
    tools=tools,
    prompt= sys_prompt_base
)



user_prompt = f"\nProcess Mining Representations: \
    1. Petri Net from inductive miner: {str(hiring_net)} \n with initial marking {str(hiring_im)}, final marking {str(hiring_fm)},\
    2. BPMN extracted from inductive miner: {hiring_bpmn_str}\n \
    3. DFG (Directly-Follows Graph) abstraction: {hiring_dfg}"

# Save experiment configuration
config = {
    "run_timestamp": run_timestamp,
    "run_folder": run_folder,
    "model": model_name,
    "tools": [tool.__name__ for tool in tools],
    "prompt_file": prompt_file,
    "hiring_net": str(hiring_net),
    "hiring_im": str(hiring_im),
    "hiring_fm": str(hiring_fm),
    "hiring_bpmn_str": hiring_bpmn_str,
    "hiring_dfg": str(hiring_dfg),
    "user_prompt": user_prompt
}
config_file = os.path.join(run_folder, "experiment_config.json")
with open(config_file, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

inputs = {"messages": [{"role": "user", "content": user_prompt}]}

# Create log file for this run
log_file = os.path.join(run_folder, "agent_logs.txt")

breakpoint()

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"{'=' * 80}\nAGENT RUN LOG\nTimestamp: {datetime.now().isoformat()}\n{'=' * 80}\n\nUSER PROMPT:\n{'-' * 80}\n{user_prompt}\n\n{'=' * 80}\nAGENT RESPONSE STREAM:\n{'=' * 80}\n\n")
    
    # Stream agent updates and log them
    for chunk in graph.stream(inputs, stream_mode="updates"):
        # Convert chunk to string representation
        chunk_str = json.dumps(chunk, indent=2, default=str) if isinstance(chunk, dict) else str(chunk)
        
        # Print to console
        print(chunk)
        
        log.write(f"[{datetime.now().isoformat()}] Chunk Update:\n{chunk_str}\n{'-' * 80}\n\n"); log.flush()
    
    log.write(f"{'=' * 80}\nEND OF RUN\nTimestamp: {datetime.now().isoformat()}\n{'=' * 80}\n")


