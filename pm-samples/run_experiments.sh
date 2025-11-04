#!/bin/bash

# Enhanced Agentic Process Miner - Experiment Runner
# This script provides easy shortcuts for common experiments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if API keys are set
check_api_keys() {
    if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        print_error "No API keys found in environment!"
        print_info "Please set one of the following:"
        print_info "  export ANTHROPIC_API_KEY='your-key'"
        print_info "  export OPENAI_API_KEY='your-key'"
        print_info "Or create a .env file with your keys"
        exit 1
    fi
}

# Load .env file if it exists
if [ -f .env ]; then
    print_info "Loading API keys from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Function to run experiment
run_experiment() {
    local dataset=$1
    local representations=$2
    local model=$3

    print_info "Running experiment:"
    print_info "  Dataset: $dataset"
    print_info "  Representations: $representations"
    print_info "  Model: $model"

    python enhanced_agentic_miner.py \
        --dataset "$dataset" \
        --representations $representations \
        --model "$model"
}

# Main menu
case "${1:-menu}" in
    list)
        print_info "Listing available datasets..."
        python enhanced_agentic_miner.py --list
        ;;

    quick)
        # Quick test with small dataset
        check_api_keys
        print_info "Running quick test on hiring dataset (DFG only)..."
        run_experiment "hiring" "dfg" "anthropic:claude-sonnet-4-5-latest"
        ;;

    full)
        # Full analysis with all representations
        check_api_keys
        dataset="${2:-hiring}"
        print_info "Running full analysis on $dataset dataset..."
        run_experiment "$dataset" "petri_net bpmn dfg heuristics_net process_tree" "anthropic:claude-sonnet-4-5-latest"
        ;;

    benchmark)
        # Run on all benchmark datasets
        check_api_keys
        print_info "Running benchmark on all datasets..."
        for dataset in hiring hospital lending renting; do
            print_info "Processing $dataset..."
            run_experiment "$dataset" "petri_net bpmn dfg" "anthropic:claude-sonnet-4-5-latest"
            sleep 5  # Rate limiting
        done
        ;;

    compare-models)
        # Compare different models on same dataset
        check_api_keys
        dataset="${2:-hiring}"
        print_info "Comparing models on $dataset dataset..."

        # Test with Claude Sonnet
        print_info "Testing Claude Sonnet..."
        run_experiment "$dataset" "petri_net bpmn" "anthropic:claude-sonnet-4-5-latest"

        # Test with GPT-4o-mini if OpenAI key is available
        if [ ! -z "$OPENAI_API_KEY" ]; then
            print_info "Testing GPT-4o-mini..."
            run_experiment "$dataset" "petri_net bpmn" "openai:gpt-4o-mini"
        fi
        ;;

    custom)
        # Custom run with all parameters
        check_api_keys
        shift  # Remove 'custom' from arguments
        python enhanced_agentic_miner.py "$@"
        ;;

    install)
        # Install dependencies
        print_info "Installing dependencies..."
        pip install -r requirements.txt
        print_info "Dependencies installed successfully!"
        ;;

    clean)
        # Clean up run directories
        print_warning "This will delete all run directories. Continue? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf runs/
            mkdir runs
            print_info "Cleaned run directories"
        else
            print_info "Cleanup cancelled"
        fi
        ;;

    menu|help|*)
        echo "Enhanced Agentic Process Miner - Experiment Runner"
        echo ""
        echo "Usage: ./run_experiments.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  list              List all available datasets"
        echo "  quick             Quick test with hiring dataset (DFG only)"
        echo "  full [dataset]    Full analysis with all representations"
        echo "  benchmark         Run on all benchmark datasets"
        echo "  compare-models    Compare different LLM models"
        echo "  custom [args]     Run with custom arguments"
        echo "  install           Install Python dependencies"
        echo "  clean             Clean up run directories"
        echo "  help              Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_experiments.sh quick"
        echo "  ./run_experiments.sh full hospital"
        echo "  ./run_experiments.sh custom --dataset lending --model openai:gpt-4o-mini"
        echo ""
        echo "Environment:"
        if [ ! -z "$ANTHROPIC_API_KEY" ]; then
            echo "  ANTHROPIC_API_KEY: Set ✓"
        else
            echo "  ANTHROPIC_API_KEY: Not set ✗"
        fi
        if [ ! -z "$OPENAI_API_KEY" ]; then
            echo "  OPENAI_API_KEY: Set ✓"
        else
            echo "  OPENAI_API_KEY: Not set ✗"
        fi
        ;;
esac