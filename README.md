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

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
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
    max_iterations=10
)

# Process long context
result = rlm.completion(
    context="Very long context...",
    query="What is the answer to the question?"
)
```

## Paper Compliance

This implementation faithfully reproduces the key innovations from the RLM paper:

1. **External Context Treatment**: Context is stored as an external variable in the REPL environment rather than being fed directly to the neural network.

2. **Programmatic Interaction**: LLMs can write code to peek into, decompose, and recursively call themselves over programmatic snippets of the context.

3. **Recursive Self-Calling**: The system enables the LLM to invoke itself recursively on smaller chunks of the context.

4. **Scalability**: The system can handle inputs orders of magnitude beyond model context windows.


## Performance Characteristics

- Handles contexts significantly beyond model context windows
- Comparable or better quality than base LLMs and common long-context scaffolds
- Comparable or cheaper cost per query compared to alternatives
- Maintains strong performance as context length and task complexity increase

## Issues and Solutions

### String Formatting Issue with Curly Braces
**Problem**: The system prompt contained code examples with `{chunk}` placeholders, which Python's `.format()` method interpreted as format placeholders, causing a KeyError when trying to format the prompt.

**Solution**: Escaped all curly braces in the code examples within the system prompt by doubling them (`{{` and `}}`) so they wouldn't be interpreted as format placeholders.

### Infinite Loop in RLM Execution
**Problem**: The RLM was getting stuck in an infinite loop, continuously generating code blocks without reaching a final answer.

**Solution**: Added proper termination condition checking in the main iteration loop to detect when a final answer was found in the REPL environment variables.

### Local Model Compatibility Issues
**Problem**: The local model wasn't following the instructions as well as expected, often generating empty responses or not using code blocks effectively.

**Solution**: Adjusted the system prompts to be more explicit and reduced the complexity of test cases to better match the capabilities of the local model.

### Context Loading and Type Handling
**Problem**: The `_convert_context` method could return both `context_data` and `context_str` as non-None values, causing the second one to overwrite the first in the REPL environment.

**Solution**: Ensured the `_convert_context` method properly returns only one of them as non-None based on the input type.

### Cost Tracking Implementation
**Problem**: The local model API doesn't provide detailed token usage, making accurate cost tracking difficult.

**Solution**: Implemented estimated cost calculation based on character counts with configurable pricing per model.

### Code Execution Security and State Management
**Problem**: The REPL environment needed to securely execute user-generated code while maintaining state across iterations.

**Solution**: Created a restricted execution environment with a safe set of built-ins and proper state isolation.

### Model Response Parsing
**Problem**: The system needed to reliably extract code blocks and final answers from model responses, but responses varied in format.

**Solution**: Used robust regex patterns and multiple parsing strategies to handle different response formats.

### API URL Configuration and Model Selection
**Problem**: The API URL was hardcoded and the model had to be specified manually, making it difficult to switch between different endpoints or automatically select available models.

**Solution**: Added support for environment variable configuration of the API URL with a default fallback, and implemented automatic model selection from the available models endpoint.

## Architecture

### External Context Storage
Storing context as an external variable in REPL rather than passing to model directly allows handling of arbitrarily long contexts.

### Two-Tier Architecture
Separating root LLM from sub-LLMs allows for different models at different levels of recursion.

### Message-Based Communication
Using structured messages between iterations maintains conversation history and context.

### Modular Design
Separating concerns into RLM base class, REPL environment, and utility functions enables reuse and testing.


## References

- Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv preprint arXiv:2512.24601.
- https://arxiv.org/pdf/2512.24601
- https://github.com/alexzhang13/rlm-minimal