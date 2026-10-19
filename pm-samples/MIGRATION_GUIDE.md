# Migration Guide: From `agentic_miner_template.py` to `enhanced_agentic_miner.py`

This guide helps you transition from the original implementation to the enhanced version.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up API keys** (CRITICAL - your keys are currently exposed!):
```bash
# First, revoke your exposed keys:
# - Go to https://console.anthropic.com/ → Revoke old key
# - Go to https://platform.openai.com/ → Revoke old key

# Generate new keys and set them:
export ANTHROPIC_API_KEY='your-new-anthropic-key'
export OPENAI_API_KEY='your-new-openai-key'

# Or create a .env file:
cp .env.example .env
# Edit .env with your new keys
```

3. **Run your first experiment**:
```bash
# Same as your original script but better:
python enhanced_agentic_miner.py --dataset hiring
```

## Key Differences

### 1. API Key Management

**Before (INSECURE)**:
```python
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-..."  # EXPOSED!
```

**After (SECURE)**:
```python
# Keys from environment or .env file
api_key = os.getenv("ANTHROPIC_API_KEY")
```

### 2. Dataset Selection

**Before**:
```python
# Hardcoded path
hiring_log = pm4py.read_xes("/home/marco/.../hiring_log_high.xes")
```

**After**:
```bash
# CLI interface
python enhanced_agentic_miner.py --dataset hiring
# Or any dataset:
python enhanced_agentic_miner.py --list  # See all available
```

### 3. Representation Configuration

**Before**:
```python
# Fixed representations
hiring_net, hiring_im, hiring_fm = pm4py.discover_petri_net_inductive(hiring_log)
hiring_bpmn = pm4py.discover_bpmn_inductive(hiring_log)
hiring_dfg = pm4py.llm.abstract_dfg(hiring_log)
```

**After**:
```bash
# Choose what you need
python enhanced_agentic_miner.py --dataset hiring --representations bpmn dfg
# Or get everything:
python enhanced_agentic_miner.py --dataset hiring --representations petri_net bpmn dfg heuristics_net process_tree
```

### 4. Interpretation Guides

**Before**:
```python
# Raw representations without context
user_prompt = f"Petri Net: {str(hiring_net)}"
```

**After**:
```python
# Each representation includes interpretation guide
"""
=== PETRI NET INTERPRETATION GUIDE ===
A Petri net is a mathematical model...
[Detailed instructions on how to read the model]

DISCOVERED MODEL:
[The actual model]
"""
```

## Feature Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| **Security** | Hardcoded keys ⚠️ | Environment variables ✅ |
| **Datasets** | Hardcoded path | CLI selection from benchmark/ |
| **Representations** | Fixed 3 types | Configurable 5+ types |
| **Interpretation** | None | Detailed guides for each type |
| **Error Handling** | None | Comprehensive try-catch |
| **Logging** | Basic | Structured with timestamps |
| **Code Structure** | Single file, procedural | Classes, modular, extensible |
| **Configuration** | None | JSON-based experiment tracking |
| **Model Selection** | Comment/uncomment | CLI parameter |
| **Output** | Single run folder | Organized by dataset + timestamp |

## Equivalent Commands

### Original workflow:
```python
# Edit agentic_miner_template.py to change dataset
# Run: python agentic_miner_template.py
# Check runs/ folder
```

### Enhanced workflow:
```bash
# List datasets
./run_experiments.sh list

# Quick test
./run_experiments.sh quick

# Full analysis
./run_experiments.sh full hospital

# Custom configuration
python enhanced_agentic_miner.py \
    --dataset hospital \
    --representations petri_net bpmn \
    --model openai:gpt-4o-mini
```

## Output Structure Comparison

### Original:
```
runs/
└── run_20251031_143022/
    ├── new_prompt.txt
    └── [agent outputs]
```

### Enhanced:
```
runs/
└── run_hospital_20251031_143022/
    ├── experiment_config.json    # Full reproducibility
    ├── system_prompt.txt          # System prompt used
    ├── user_prompt.txt            # Generated prompt with guides
    ├── agent_logs.txt             # Detailed streaming logs
    ├── AsIsProcess.bpmn           # Generated models
    ├── IdealProcess.bpmn
    └── process_analysis.txt
```

## Advanced Features

### 1. Experiment Comparison
```bash
# Run same dataset with different models
./run_experiments.sh compare-models hospital

# Results will be in separate timestamped folders
# Easy to compare outputs
```

### 2. Custom XES Files
```bash
# Not limited to benchmark folder
python enhanced_agentic_miner.py --xes-file /path/to/custom.xes
```

### 3. Batch Processing
```bash
# Process all benchmarks
./run_experiments.sh benchmark
```

## Code Extension Example

To add a new representation type:

```python
class AlphaMinerRepresentation(ProcessRepresentation):
    """Alpha Miner representation."""

    def discover(self):
        self.representation = pm4py.discover_petri_net_alpha(self.log)
        return self.representation

    def to_string(self) -> str:
        # Convert to string
        return str(self.representation)

    def get_interpretation_guide(self) -> str:
        return """
        === ALPHA MINER INTERPRETATION GUIDE ===
        The Alpha algorithm discovers Petri nets from event logs...
        """

# Register in main class
REPRESENTATION_CLASSES[RepresentationType.ALPHA] = AlphaMinerRepresentation
```

## Troubleshooting

### Issue: "Module pm4py not found"
```bash
pip install -r requirements.txt
```

### Issue: "ANTHROPIC_API_KEY not found"
```bash
export ANTHROPIC_API_KEY='your-key'
# Or use .env file
```

### Issue: Want to use the old script
Both scripts can coexist:
- `agentic_miner_template.py` - Original (update API keys first!)
- `enhanced_agentic_miner.py` - Enhanced version

## Benefits of Migration

1. **Security**: No exposed API keys
2. **Flexibility**: Configure representations per run
3. **Understanding**: LLM gets interpretation guides
4. **Reproducibility**: Complete experiment tracking
5. **Scalability**: Easy to run on all datasets
6. **Maintainability**: Clean, modular code
7. **Debugging**: Comprehensive logging

## Next Steps

1. **Immediate**: Revoke and replace exposed API keys
2. **Setup**: Install dependencies and configure .env
3. **Test**: Run `./run_experiments.sh quick`
4. **Explore**: Try different representations and models
5. **Extend**: Add custom representations or tools

The enhanced version maintains full compatibility with your workflow while adding production-ready features and security. All your existing prompts and workflows will work with improved reliability and observability.