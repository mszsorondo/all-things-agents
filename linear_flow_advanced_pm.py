from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Dict, Any, List, Literal, TypedDict, Optional
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
# Note: decorators might not be available in all langfuse versions
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    # Create dummy decorators if not available
    def observe(name=None):
        def decorator(func):
            return func
        return decorator

    class DummyContext:
        def update_current_observation(self, metadata):
            pass
        def get_current_observation_id(self):
            return "dummy_obs_id"
        def get_current_trace_id(self):
            return "dummy_trace_id"

    langfuse_context = DummyContext()

load_dotenv()

# Add process-mining to path
sys.path.append(str(Path(__file__).parent / "process-mining"))
from process_mining_logger import ProcessMiningLogger

# Langfuse setup
langfuse = get_client()
langfuse_handler = CallbackHandler()

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Create experiment directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = Path("experiments") / f"run_{timestamp}"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Created experiment directory: {EXPERIMENT_DIR}")

# Initialize Process Mining Logger
PM_LOGGER = ProcessMiningLogger(
    case_id=f"run_{timestamp}",
    architecture="linear_flow",
    problem="generic_python_task",
    experiment_dir=EXPERIMENT_DIR
)

# ==================== SHARED STATE ====================

class AgentState(MessagesState):
    """Extended state to track agent-specific information."""
    current_agent: str
    code_files: List[str]
    test_files: List[str]
    test_results: Dict[str, Any]
    specification: str

# ==================== CODING AGENT TOOLS ====================

@tool
def write_code(filename: str, code: str) -> str:
    """Write Python code to a file. Use for creating implementation files."""
    try:
        # Ensure it's a .py file
        if not filename.endswith('.py'):
            filename += '.py'

        filepath = EXPERIMENT_DIR / filename
        with open(filepath, "w") as f:
            f.write(code)

        print(f"✅ Created file: {filepath}")
        return f"Successfully created {filename} with {len(code.splitlines())} lines of code"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool
def edit_code(filename: str, old_code: str, new_code: str) -> str:
    """Edit existing Python code by replacing a section."""
    try:
        filepath = EXPERIMENT_DIR / filename
        with open(filepath, "r") as f:
            content = f.read()

        if old_code in content:
            content = content.replace(old_code, new_code)
            with open(filepath, "w") as f:
                f.write(content)
            return f"Successfully edited {filename}"
        else:
            return f"Could not find the specified code section in {filename}"
    except FileNotFoundError:
        return f"File {filename} not found"
    except Exception as e:
        return f"Error editing file: {str(e)}"

@tool
def read_code(filename: str) -> str:
    """Read Python code from a file."""
    try:
        filepath = EXPERIMENT_DIR / filename
        with open(filepath, "r") as f:
            content = f.read()
        return f"Content of {filename}:\n```python\n{content}\n```"
    except FileNotFoundError:
        return f"File {filename} not found in {EXPERIMENT_DIR}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ==================== TESTING AGENT TOOLS ====================

@tool
def write_tests(filename: str, test_code: str) -> str:
    """Write test code to a file. File should follow test naming conventions."""
    try:
        # Ensure proper test file naming
        if not (filename.startswith("test_") or filename.endswith("_test.py")):
            if not filename.startswith("test_"):
                filename = "test_" + filename

        if not filename.endswith('.py'):
            filename += '.py'

        filepath = EXPERIMENT_DIR / filename
        with open(filepath, "w") as f:
            f.write(test_code)

        print(f"🧪 Created test file: {filepath}")
        return f"Successfully created test file {filename} with {len(test_code.splitlines())} lines"
    except Exception as e:
        return f"Error writing test file: {str(e)}"

@tool
def run_tests(test_file: str) -> str:
    """Execute tests and return detailed results."""
    try:
        # Change to experiment directory for proper imports
        original_cwd = os.getcwd()
        os.chdir(EXPERIMENT_DIR)

        # Try pytest first
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if "No module named pytest" in result.stderr:
            # Fallback to unittest
            test_module = test_file.replace(".py", "")
            result = subprocess.run(
                ["python", "-m", "unittest", test_module, "-v"],
                capture_output=True,
                text=True,
                timeout=30
            )

        # Change back to original directory
        os.chdir(original_cwd)

        output = f"Test Results for {test_file}:\n"
        output += f"Exit Code: {result.returncode}\n"
        output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        if result.returncode == 0:
            output += "\n✅ All tests passed!"
        else:
            output += "\n❌ Some tests failed. Please review and fix."

        print(f"🏃 Ran tests: {test_file}")
        return output
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        return "❌ Test execution timed out after 30 seconds"
    except Exception as e:
        os.chdir(original_cwd)
        return f"❌ Error running tests: {str(e)}"

@tool
def run_code(filename: str, args: str = "") -> str:
    """Execute Python code file directly to test functionality."""
    try:
        cmd = ["python", filename]
        if args:
            cmd.extend(args.split())

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = f"Execution of {filename}:\n"
        output += f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"Output:\n{result.stdout}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr}\n"

        return output
    except subprocess.TimeoutExpired:
        return "Execution timed out after 10 seconds"
    except Exception as e:
        return f"Error running code: {str(e)}"

# ==================== SPEC AGENT TOOLS ====================

@tool
def analyze_implementation(spec_requirements: str, implementation_file: str) -> str:
    """Analyze if implementation meets specification requirements."""
    try:
        with open(implementation_file, "r") as f:
            code = f.read()

        analysis = f"Analysis of {implementation_file} against specification:\n\n"

        # Parse requirements (simple keyword matching for demo)
        requirements = spec_requirements.lower().split(',')

        found_features = []
        missing_features = []

        for req in requirements:
            req = req.strip()
            if req in code.lower():
                found_features.append(req)
            else:
                missing_features.append(req)

        if found_features:
            analysis += f"✅ Found features: {', '.join(found_features)}\n"
        if missing_features:
            analysis += f"⚠️ Potentially missing: {', '.join(missing_features)}\n"

        # Check for error handling
        if "try" in code and "except" in code:
            analysis += "✅ Error handling present\n"
        else:
            analysis += "⚠️ Consider adding error handling\n"

        # Check for documentation
        if '"""' in code or "'''" in code:
            analysis += "✅ Documentation present\n"
        else:
            analysis += "⚠️ Consider adding docstrings\n"

        return analysis
    except Exception as e:
        return f"Error analyzing implementation: {str(e)}"

# ==================== AGENT NODES ====================

@observe(name="spec_agent")
def create_spec_agent(state: AgentState) -> Dict[str, Any]:
    """Specification agent that creates detailed requirements for any Python task."""
    PM_LOGGER.log_agent_start("spec_agent", "gpt-4o-mini")

    spec_tools = [analyze_implementation]
    spec_model = model.bind_tools(spec_tools)

    # Extract the task from the user message
    user_message = state["messages"][0].content if state["messages"] else "No task specified"

    system_prompt = f"""You are a specification agent responsible for creating clear, detailed specifications.

    The user has requested: {user_message[:500]}

    Your task:
    1. Analyze the user's requirements carefully
    2. Create a detailed specification with:
       - Clear functional requirements (list all functions/methods needed)
       - Input/output specifications for each function
       - Error handling requirements
       - Edge cases to consider
       - Performance considerations if relevant
    3. Be thorough and precise - other agents will implement based solely on your specification
    4. For any programming task, specify the expected function names and their behavior

    Create a comprehensive specification that any developer could implement from."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"][:1]

    # Track token usage
    response = spec_model.invoke(messages, config={"callbacks": [langfuse_handler]})

    # Extract specification from response
    spec_content = response.content if hasattr(response, 'content') else str(response)

    # Log completion with token metrics
    tokens_in = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0) if hasattr(response, 'response_metadata') else 0
    tokens_out = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0) if hasattr(response, 'response_metadata') else 0

    PM_LOGGER.log_agent_complete(
        "spec_agent",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=True
    )

    # Log transition to next agent
    PM_LOGGER.log_transition("spec_agent", "coding_agent", "specification_complete")

    return {
        "messages": [response],
        "specification": spec_content,
        "current_agent": "spec_agent"
    }

@observe(name="coding_agent")
def create_coding_agent(state: AgentState) -> Dict[str, Any]:
    """Coding agent that implements based on specifications."""
    PM_LOGGER.log_agent_start("coding_agent", "gpt-4o-mini")

    coding_tools = [write_code, edit_code, read_code]
    coding_model = model.bind_tools(coding_tools)

    # Get the specification from state
    specification = state.get('specification', 'No specification provided')

    system_prompt = f"""You are a coding agent responsible for implementing Python solutions.

    Specification to implement:
    {specification}

    Your tasks:
    1. Create a Python implementation that meets ALL requirements in the specification
    2. Use the write_code tool to create appropriate Python files
    3. Implement all functions/methods mentioned in the specification
    4. Include proper error handling as specified
    5. Make the code clean, well-structured, and properly documented
    6. Name the main file appropriately based on the task (e.g., 'calculator.py', 'string_utils.py', etc.)

    Create a complete, working implementation based on the specification."""

    # Only include the initial human message for context
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    messages = [SystemMessage(content=system_prompt)] + human_messages[:1]

    response = coding_model.invoke(messages, config={"callbacks": [langfuse_handler]})

    # Track created files and tools used
    code_files = state.get("code_files", [])
    tools_used = []

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tools_used.append(tool_name)

            # Log tool invocation
            PM_LOGGER.log_tool_invocation(
                tool_name=tool_name,
                agent_name="coding_agent",
                parameters=tool_call.get('args', {})
            )

            # Track created files
            if tool_name == 'write_code':
                filename = tool_call.get('args', {}).get('filename', 'unknown.py')
                if filename not in code_files:
                    code_files.append(filename)

    # Log completion
    tokens_in = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0) if hasattr(response, 'response_metadata') else 0
    tokens_out = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0) if hasattr(response, 'response_metadata') else 0

    PM_LOGGER.log_agent_complete(
        "coding_agent",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tools_used=tools_used,
        success=True
    )

    # Log transition
    PM_LOGGER.log_transition("coding_agent", "testing_agent", "code_complete")

    return {
        "messages": [response],
        "code_files": code_files,
        "current_agent": "coding_agent"
    }

@observe(name="testing_agent")
def create_testing_agent(state: AgentState) -> Dict[str, Any]:
    """Testing agent that creates and runs comprehensive tests."""
    PM_LOGGER.log_agent_start("testing_agent", "gpt-4o-mini")

    testing_tools = [write_tests, run_tests, read_code, run_code]
    testing_model = model.bind_tools(testing_tools)

    code_files = state.get("code_files", [])
    if not code_files:
        code_files = ["implementation.py"]  # Default name if none found

    # Initialize conversation for testing agent
    messages = []
    tools_used = []
    total_tokens_in = 0
    total_tokens_out = 0

    # Step 1: Read the implementation
    primary_file = code_files[0] if code_files else "implementation.py"
    read_prompt = f"""Read the {primary_file} file to understand the implementation."""
    messages = [SystemMessage(content=read_prompt)]

    read_response = testing_model.invoke(messages, config={"callbacks": [langfuse_handler]})
    messages.append(read_response)

    # Track tokens
    if hasattr(read_response, 'response_metadata'):
        total_tokens_in += read_response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
        total_tokens_out += read_response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)

    # Execute the read tool if it was called
    if hasattr(read_response, "tool_calls") and read_response.tool_calls:
        for tool_call in read_response.tool_calls:
            tools_used.append(tool_call.get('name', 'unknown'))
            PM_LOGGER.log_tool_invocation(
                tool_name=tool_call.get('name', 'unknown'),
                agent_name="testing_agent",
                parameters=tool_call.get('args', {})
            )
        tool_result = testing_tool_node.invoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # Step 2: Write the tests based on what was read
    test_filename = primary_file.replace('.py', '')  # Remove .py extension for test naming

    write_prompt = f"""Based on the {primary_file} implementation you just read:

    Write comprehensive unit tests using the write_tests tool with:
    - filename='{test_filename}' (the tool will add test_ prefix)
    - test_code containing:
      * import unittest
      * Import all functions/classes from {primary_file}
      * A Test class with test methods covering:
        - Normal cases with expected outputs
        - Edge cases (empty inputs, boundary values, etc.)
        - Error cases matching EXACT error messages from the implementation
      * Use appropriate assertions (assertEqual, assertTrue, assertRaises, etc.)

    Create thorough tests that validate all functionality."""

    messages.append(SystemMessage(content=write_prompt))
    write_response = testing_model.invoke(messages, config={"callbacks": [langfuse_handler]})
    messages.append(write_response)

    # Track tokens
    if hasattr(write_response, 'response_metadata'):
        total_tokens_in += write_response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
        total_tokens_out += write_response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)

    # Execute the write tool if it was called
    test_file_created = f"test_{test_filename}.py"
    if hasattr(write_response, "tool_calls") and write_response.tool_calls:
        for tool_call in write_response.tool_calls:
            tools_used.append(tool_call.get('name', 'unknown'))
            PM_LOGGER.log_tool_invocation(
                tool_name=tool_call.get('name', 'unknown'),
                agent_name="testing_agent",
                parameters=tool_call.get('args', {})
            )
        tool_result = testing_tool_node.invoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # Step 3: Run the tests
    run_prompt = f"""Now run the tests you just created using run_tests('{test_file_created}')"""
    messages.append(SystemMessage(content=run_prompt))
    run_response = testing_model.invoke(messages, config={"callbacks": [langfuse_handler]})
    messages.append(run_response)

    # Track tokens
    if hasattr(run_response, 'response_metadata'):
        total_tokens_in += run_response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
        total_tokens_out += run_response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)

    # Execute the run tool if it was called
    if hasattr(run_response, "tool_calls") and run_response.tool_calls:
        for tool_call in run_response.tool_calls:
            tools_used.append(tool_call.get('name', 'unknown'))
            PM_LOGGER.log_tool_invocation(
                tool_name=tool_call.get('name', 'unknown'),
                agent_name="testing_agent",
                parameters=tool_call.get('args', {})
            )
        tool_result = testing_tool_node.invoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # Parse test results from the last message
    last_message = messages[-1] if messages else None
    test_results = {
        "passed": "✅" in str(last_message.content) if last_message and hasattr(last_message, 'content') else False,
        "feedback": str(last_message.content) if last_message and hasattr(last_message, 'content') else ""
    }

    # Track test files
    test_files = state.get("test_files", [])
    if test_file_created not in test_files:
        test_files.append(test_file_created)

    # Log agent completion
    PM_LOGGER.log_agent_complete(
        "testing_agent",
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        tools_used=tools_used,
        success=test_results["passed"]
    )

    # Log transition
    PM_LOGGER.log_transition("testing_agent", "review_agent", "tests_complete")

    # Create a summary message
    summary = f"""Testing completed. {"✅ All tests passed!" if test_results["passed"] else "❌ Some tests failed."}

{test_results["feedback"]}"""

    summary_message = AIMessage(content=summary)

    return {
        "messages": [summary_message],
        "test_files": test_files,
        "test_results": test_results,
        "current_agent": "testing_agent"
    }

@observe(name="review_agent")
def create_review_agent(state: AgentState) -> Dict[str, Any]:
    """Review agent that provides final assessment."""
    PM_LOGGER.log_agent_start("review_agent", "gpt-4o-mini")

    review_prompt = f"""You are a review agent providing the FINAL assessment.
    The pipeline has completed and will not loop back regardless of issues found.

    Review the complete pipeline:
    - Specification created: {bool(state.get('specification'))}
    - Code files created: {state.get('code_files', [])}
    - Test files created: {state.get('test_files', [])}
    - Tests passed: {state.get('test_results', {}).get('passed', 'Unknown')}

    Provide a concise summary of:
    1. What was successfully implemented
    2. Test coverage and results
    3. Any errors or failures encountered
    4. What would need to be fixed in future runs (but will NOT be fixed now)

    Be brief but comprehensive. This is the END of the pipeline."""

    # Review agent doesn't need tools, just assessment
    messages = [SystemMessage(content=review_prompt)]
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})

    # Log completion
    tokens_in = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0) if hasattr(response, 'response_metadata') else 0
    tokens_out = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0) if hasattr(response, 'response_metadata') else 0

    PM_LOGGER.log_agent_complete(
        "review_agent",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=True
    )

    return {
        "messages": [response],
        "current_agent": "review_agent"
    }

# ==================== TOOL NODES ====================

spec_tool_node = ToolNode([analyze_implementation])
coding_tool_node = ToolNode([write_code, edit_code, read_code])
testing_tool_node = ToolNode([write_tests, run_tests, read_code, run_code])

# ==================== ROUTING FUNCTIONS ====================

def should_continue_to_tools(state: AgentState) -> str:
    """Check if we should continue to tools or next agent."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # Check if the last message has tool calls
    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "continue"

def route_from_testing(state: AgentState) -> Literal["review_agent"]:
    """Always proceed to review agent - no loops."""
    return "review_agent"

# ==================== BUILD THE GRAPH ====================

builder = StateGraph(AgentState)

# Add agent nodes
builder.add_node("spec_agent", create_spec_agent)
builder.add_node("spec_tools", spec_tool_node)
builder.add_node("coding_agent", create_coding_agent)
builder.add_node("coding_tools", coding_tool_node)
builder.add_node("testing_agent", create_testing_agent)
builder.add_node("testing_tools", testing_tool_node)
builder.add_node("review_agent", create_review_agent)

# Add edges for spec agent
builder.add_edge(START, "spec_agent")
builder.add_conditional_edges(
    "spec_agent",
    should_continue_to_tools,
    {"tools": "spec_tools", "continue": "coding_agent"}
)
# After spec tools execute, go to coding agent (not back to spec)
builder.add_edge("spec_tools", "coding_agent")

# Add edges for coding agent
builder.add_conditional_edges(
    "coding_agent",
    should_continue_to_tools,
    {"tools": "coding_tools", "continue": "testing_agent"}
)
# After coding tools execute, go to testing agent (not back to coding)
builder.add_edge("coding_tools", "testing_agent")

# Add edges for testing agent
builder.add_conditional_edges(
    "testing_agent",
    should_continue_to_tools,
    {"tools": "testing_tools", "continue": "review_agent"}
)
# Testing agent no longer loops back since we handle everything internally
builder.add_edge("testing_tools", "review_agent")

# Review agent ends the flow
builder.add_edge("review_agent", END)

# Compile the graph
graph = builder.compile()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("Starting Enhanced Multi-Agent System with Tools...")
    print("="*60)

    # Initial task - this can be any Python programming task
    initial_message = HumanMessage(
        content="Create a calculator with add, subtract, multiply, and divide functions. "
                "Include error handling for division by zero and invalid inputs."
    )

    # Log the user request to PM logger
    PM_LOGGER.log_user_request(initial_message.content)

    # Run the graph
    try:
        result = graph.invoke(
            {"messages": [initial_message]},
            config={"callbacks": [langfuse_handler]},
        )

        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)

        # Summary
        success = result.get('test_results', {}).get('passed', False)
        print(f"\n📋 Specification: {'✅' if result.get('specification') else '❌'}")
        print(f"📝 Code Files: {result.get('code_files', [])}")
        print(f"🧪 Test Files: {result.get('test_files', [])}")
        print(f"✅ Tests Passed: {result.get('test_results', {}).get('passed', 'Unknown')}")

        # Finalize PM log
        PM_LOGGER.finalize(success=success)

        # Save detailed results
        result_path = EXPERIMENT_DIR / "advanced_result.json"
        with open(result_path, "w") as f:
            json.dump({
                "specification": result.get("specification", ""),
                "code_files": result.get("code_files", []),
                "test_files": result.get("test_files", []),
                "test_results": result.get("test_results", {}),
                "messages": [str(msg) for msg in result["messages"]]
            }, f, indent=2)

        print(f"\n📊 Full results saved to {result_path}")

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        PM_LOGGER.finalize(success=False, error=str(e))
        import traceback
        traceback.print_exc()