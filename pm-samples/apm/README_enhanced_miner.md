# Enhanced Agentic Process Miner

A production-ready, modular system for process mining with LLM-based interpretation. This tool combines classical process mining algorithms with large language models to automatically discover, analyze, and optimize business processes.

## Features

- **Multiple Process Representations**: Supports Petri nets, BPMN, DFG, Heuristics nets, and Process trees
- **Interpretation Guides**: Each representation includes detailed interpretation instructions for the LLM
- **Configurable Pipeline**: Select which representations to use for analysis
- **Production-Ready**: Proper error handling, logging, and configuration management
- **Security-First**: API keys via environment variables, no hardcoded credentials
- **Comprehensive Output**: Generates AS-IS models, IDEAL models, and analysis documentation

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up API keys**:
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual API keys
nano .env

# Or export directly in your shell
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'
```

## Usage

### List Available Datasets
```bash
python enhanced_agentic_miner.py --list
```

### Run on a Specific Dataset
```bash
# Run on hiring dataset with default representations
python enhanced_agentic_miner.py --dataset hiring

# Run on hospital dataset with specific representations
python enhanced_agentic_miner.py --dataset hospital --representations petri_net bpmn dfg

# Use GPT-4 instead of Claude
python enhanced_agentic_miner.py --dataset lending --model openai:gpt-4o-mini
```

### Run on Custom XES File
```bash
python enhanced_agentic_miner.py --xes-file /path/to/your/event_log.xes
```

### Advanced Options
```bash
# All available representations
python enhanced_agentic_miner.py --dataset hiring \
    --representations petri_net bpmn dfg heuristics_net process_tree

# Custom output directory
python enhanced_agentic_miner.py --dataset hospital \
    --output-dir experiments/hospital_analysis

# Disable logging for cleaner output
python enhanced_agentic_miner.py --dataset renting --no-logging

# Use custom system prompt
python enhanced_agentic_miner.py --dataset lending \
    --prompt-file custom_prompt.txt
```

## Available Representations

### 1. Petri Net (`petri_net`)
- Mathematical model with places and transitions
- Shows concurrent behavior and synchronization
- Best for: Formal analysis, soundness checking

### 2. BPMN (`bpmn`)
- Business-friendly graphical notation
- Industry standard for process documentation
- Best for: Communication with stakeholders

### 3. Directly-Follows Graph (`dfg`)
- Shows frequency of activity succession
- Simple and intuitive visualization
- Best for: Quick overview, bottleneck detection

### 4. Heuristics Net (`heuristics_net`)
- Improved DFG with dependency measures
- Better noise handling
- Best for: Real-world logs with noise

### 5. Process Tree (`process_tree`)
- Hierarchical process structure
- Guarantees soundness
- Best for: Understanding control flow patterns

## Output Structure

Each run creates a timestamped folder containing:

```
runs/
└── run_hiring_20251101_143022/
    ├── experiment_config.json    # Full configuration
    ├── system_prompt.txt          # System prompt used
    ├── user_prompt.txt            # Generated user prompt
    ├── agent_logs.txt             # Streaming agent responses
    ├── AsIsProcess.bpmn           # Discovered AS-IS model
    ├── IdealProcess.bpmn          # Optimized IDEAL model
    └── process_analysis.txt       # Comprehensive analysis
```

## Architecture

```python
# Main Classes
AgenticProcessMiner           # Orchestrator class
├── MiningConfiguration       # Configuration management
├── ProcessRepresentation     # Abstract base for representations
│   ├── PetriNetRepresentation
│   ├── BPMNRepresentation
│   ├── DFGRepresentation
│   ├── HeuristicsNetRepresentation
│   └── ProcessTreeRepresentation
└── LLM Agent (via LangGraph)
```

## Key Features Explained

### 1. Interpretation Guides
Each representation includes detailed instructions on how to read and interpret the model. This helps the LLM understand the semantics of different process notations.

### 2. Configurable Representations
You can choose which representations to include in the analysis. This allows you to:
- Reduce token usage for cost optimization
- Focus on specific aspects of the process
- Experiment with different combinations

### 3. AS-IS vs IDEAL Models
- **AS-IS**: Reflects the actual observed behavior, including inefficiencies
- **IDEAL**: Represents the optimized target process without issues

### 4. Production Features
- Proper logging with timestamps
- Error handling and recovery
- Configuration management
- Experiment tracking
- No hardcoded credentials

## Model Selection

### Anthropic Claude Models
- `anthropic:claude-sonnet-3-5-latest` (default, balanced)
- `anthropic:claude-opus-3` (most capable, expensive)
- `anthropic:claude-haiku-3` (fastest, cheapest)

### OpenAI GPT Models
- `openai:gpt-4o` (most capable)
- `openai:gpt-4o-mini` (cost-effective)
- `openai:gpt-3.5-turbo` (fastest, cheapest)

## Troubleshooting

### Module Not Found Error
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

### API Key Error
```bash
# Check if environment variable is set
echo $ANTHROPIC_API_KEY

# Set it if missing
export ANTHROPIC_API_KEY='your-key-here'
```

### Large Event Logs
For very large event logs (>100MB), consider:
- Using sampling: Process a subset of traces
- Increasing memory: `export PYTHONHASHSEED=0`
- Using simpler representations (DFG only)

## Comparison with Original Implementation

| Feature | Original | Enhanced |
|---------|----------|----------|
| Code Structure | Single file, procedural | Class-based, modular |
| API Keys | Hardcoded ⚠️ | Environment variables ✅ |
| Representations | Fixed set | Configurable |
| Interpretation | None | Detailed guides |
| Error Handling | None | Comprehensive |
| Logging | Basic | Structured with timestamps |
| Dataset Selection | Hardcoded path | CLI interface |
| Configuration | None | JSON-based |
| Reproducibility | Limited | Full experiment tracking |

## Contributing

To add a new representation:

1. Create a new class inheriting from `ProcessRepresentation`
2. Implement `discover()`, `to_string()`, and `get_interpretation_guide()`
3. Register it in `REPRESENTATION_CLASSES`
4. Add to `RepresentationType` enum

## License

[Your License Here]

## Citation

If you use this tool in research, please cite:
```bibtex
@software{enhanced_agentic_miner,
  title = {Enhanced Agentic Process Miner},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/enhanced-agentic-miner}
}
```