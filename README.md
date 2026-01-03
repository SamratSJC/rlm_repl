# Recursive Language Models (RLMs) Implementation

This repository contains a proof of concept implementation of Recursive Language Models (RLMs) based on the paper "Recursive Language Models" by Alex L. Zhang, Tim Kraska, and Omar Khattab (https://arxiv.org/pdf/2512.24601). This implementation is also inspired by code in https://github.com/alexzhang13/rlm-minimal.

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Paper Compliance](#paper-compliance)
- [Benchmark Tasks](#benchmark-tasks)
- [Performance Characteristics](#performance-characteristics)
- [Issues and Solutions](#issues-and-solutions)
- [Architecture](#architecture)

## Overview

This implementation provides Recursive Language Model (RLM) system that allows LLMs to process arbitrarily long prompts through inference-time scaling by treating the prompt as part of an external environment. The system enables LLMs to programmatically examine, decompose, and recursively call themselves over snippets of the prompt.

## Core Components

### 1. RLM Base Class (`rlm/rlm.py`)
- Abstract base class defining the RLM interface
- Methods: `completion()`, `cost_summary()`, `reset()`

### 2. RLM with REPL (`rlm/rlm_repl.py`)
- Main RLM implementation using REPL environment
- Context stored externally in REPL, not passed to model directly
- Support for both root_model and sub_model
- Cost tracking for root and sub-LLM calls
- Iterative interaction loop with termination conditions

### 3. REPL Environment (`rlm/repl.py`)
- Python execution sandbox
- `llm_query(prompt)` function for recursive calls
- State persistence across iterations
- Output capture (stdout/stderr)
- Variable management for intermediate results

### 4. Utilities (`rlm/utils/`)
- `llm.py`: LLM client wrapper with cost tracking
- `prompts.py`: System prompts from paper Appendix D
- `tracing.py`: Detailed logging and tracing system

## Key Features

### Context Management
- Context stored as external variable in REPL environment
- Metadata provided to LLM about context size and structure
- Enables processing of contexts beyond model context window

### Recursive Sub-Calls
- `llm_query` function allows recursive LLM calls
- Separate tracking of root and sub-LLM costs
- Enables chunking and processing of large contexts

### Code Execution
- Python code execution in REPL environment
- Ability to examine, filter, and decompose context programmatically
- State persistence across iterations

### Final Answer Handling
- `FINAL()` and `FINAL_VAR()` functions for terminating responses
- Proper termination condition checking

### Cost Tracking
- Separate tracking of root and sub-LLM costs
- Token counting for both input and output
- Call counting for analysis

### Configurable API Endpoint
- Support for `RLM_API_URL` environment variable
- Automatic model selection from available models endpoint
- Default fallback to `http://localhost:8080/v1`

### Improved Configuration Options
- `max_iterations`: Maximum number of root LLM iterations before timeout (default: 20)
- `max_output_length`: Maximum length of REPL output before truncation (default: 500,000 chars)
- When max iterations reached, returns None instead of forcing an answer to align with paper's natural convergence
- Increased output length limit to reduce impact on long-output tasks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fullstackwebdev/rlm_repl
cd rlm_repl
```

2. Install dependencies (if any):
```bash
# This implementation uses standard Python libraries
# No additional dependencies required
```

3. Set up your local LLM server (e.g., using llama.cpp server, vLLM, etc.)

## Configuration

The implementation supports configurable API endpoints:

1. Set the `RLM_API_URL` environment variable to point to your LLM server:
```bash
export RLM_API_URL="http://your-llm-server:port/v1"
```

2. If not set, the system will default to `http://localhost:8080/v1`

3. The system will automatically detect available models from the `/models` endpoint and use the first available model if none is specified.

## Usage

### Basic Usage
```python
from rlm.rlm_repl import RLM_REPL

# Create RLM instance
rlm = RLM_REPL(
    model="auto",  # Automatically selects first available model
    recursive_model="auto",  # Automatically selects first available model
    max_iterations=10
)

# Process long context
result = rlm.completion(
    context="Very long context...",
    query="What is the answer to the question?"
)

# Get cost summary
costs = rlm.cost_summary()
print(f"Total cost: ${costs['total_cost']:.4f}")
```

### With Specific Model
```python
from rlm.rlm_repl import RLM_REPL

# Create RLM instance with specific model
rlm = RLM_REPL(
    model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
    recursive_model="Qwen3-Coder-REAP-25B-A3B.Q5_K_M.gguf",
    max_iterations=10,
    max_output_length=500000  # Characters before truncation
)

# Process long context
result = rlm.completion(
    context="Very long context...",
    query="What is the answer to the question?"
)

# Note: result may be None if max_iterations reached without finding final answer
if result is None:
    print("RLM reached max iterations without finding a final answer")
else:
    print(f"Result: {result}")
```


## Performance Characteristics

- Handles contexts significantly beyond model context windows
- Comparable or better quality than base LLMs and common long-context scaffolds
- Comparable or cheaper cost per query compared to alternatives
- Maintains strong performance as context length and task complexity increase


## References

- Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv preprint arXiv:2512.24601.
- https://arxiv.org/pdf/2512.24601
- https://github.com/alexzhang13/rlm-minimal