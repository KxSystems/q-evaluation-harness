# Q Evaluation Harness

**An open-source framework by KX for evaluating Large Language Models on Q/kdb+ code generation tasks.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This project introduces the first standardized evaluation benchmark for Q/kdb+, addressing a critical gap in assessing Large Language Models (LLMs) for this specialized programming language. The lack of such a benchmark has limited meaningful measurement and progress in Q code generation.

Our evaluation harness provides a robust and rigorous framework, beginning with a Q-language adaptation of OpenAI’s HumanEval. Our roadmap includes integrating additional benchmarks, such as a Q-language port of MBPP, empowering the community to effectively evaluate and advance LLM capabilities for Q.

## Model Leaderboard

Track the performance of Large Language Models on Q/kdb+ code generation tasks using our standardized evaluation framework.

| Rank | Model | Pass@1 | Pass@5 | Pass@10 |
|------|-------|--------|--------|---------|
| 🥇 | Grok 4 | **43.37%** | 68.45% | 74.32% |
| 🥈 | Claude 4 Sonnet | 37.70% | 53.47% | 59.13% | 
| 🥉 | Gemini 2.5 pro | 27.75% | 51.41% | 59.68% | 

> 📈 **[View Complete Leaderboard →](https://github.com/KxSystems/q-evaluation-harness/blob/main/docs/leaderboard.md)**  
> *See full results, historical data, and detailed analysis*

---

## Features

- 🚀 **Simple CLI**: One-command evaluation with `qeval run <dataset> <model>`.
- 📊 **Q-HumanEval Dataset**: 164 hand-crafted Q programming problems.
- 🔧 **Multi-Model Support**: Supports both closed-source APIs and open-source Hugging Face models.
- 📈 **Standard Metrics**: Pass@1, Pass@5, Pass@10 with isolated execution.
- ⏱️ **Timeout Protection**: Code execution with configurable timeout limits.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/KxSystems/q-evaluation-harness.git
cd q-evaluation-harness

# Install dependencies with Poetry
poetry install

# Activate the Poetry environment (Poetry 2.0+)
eval $(poetry env activate)
```

> **Requirements**: Python 3.10+, Poetry, and a kdb+ license.

> ⚠️ **Security Warning**: This tool executes generated code in your local environment. While we provide timeout protection, the execution is **not sandboxed**. Only run evaluations with trusted models and in isolated environments. Sandboxed execution is planned for future releases.

---

## Setup & Configuration

### Mandatory: KDB Setup

1. **Install KDB with PyKX license**: Standard install [KDB](https://kx.com/kdb-insights-personal-edition-license-download/) or follow [KDB-X](https://developer.kx.com/products/kdb-x/install)
2. **For multithreaded execution**: Set `PYKX_THREADING=1` in your environment

> 🚧 **Future**: We plan to support MCP/REST API for Q execution to remove the PyKX dependency requirement.

### Optional: API Keys

> 💡 **Note**: API keys are only needed for proprietary models (e.g., from OpenAI, Anthropic). **You can skip this if you are using open-source Hugging Face models.**

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```



---

## Quick Start

Verify your installation by running an evaluation on an open-source model. This command should work without any API keys configured.

```bash
# Make sure you're in the Poetry environment (Poetry 2.0+)
eval $(poetry env activate)

# Run evaluation on an open-source model from Hugging Face
qeval run q-humaneval Qwen/Qwen2-1.5B-Instruct
```

---

## Usage Guide: Running Evaluations

Use the `run` command to evaluate models using our standardized framework. This generates and executes Q code solutions in one step.

```bash
# Evaluate GPT-4o on Q-HumanEval (default: 50 samples for statistical significance)
qeval run q-humaneval gpt-4.1

# Evaluate an open-source model
qeval run q-humaneval google/gemma-3-4b-it

# Specify custom sample size (50 samples recommended for leaderboard submissions)
qeval run q-humaneval your-model --num-samples 50
```

> 📊 **Evaluation Standard**: Use 50 samples per problem for statistically significant results and leaderboard submissions.

---

## Submission Guidelines

Help us grow the leaderboard! Submit your model evaluation results to contribute to the Q/kdb+ AI development community.

> 📋 **[Complete Submission Guide →](docs/submission_guide.md)**

> 💬 **[Questions? Use GitHub Issues →](https://github.com/kxsystems/q-evaluation-harness/issues)**

---

## Project Reference

### Dataset: Q-HumanEval
Our primary benchmark adapts HumanEval to Q/kdb+, featuring **164 problems** with hand-verified solutions and comprehensive test cases.

### Contributing
We welcome contributions! Key areas include new datasets, model integrations, and evaluation metrics.
```bash
# Development setup
poetry install --with dev
pre-commit install
# Run tests
poetry run pytest
```

### Roadmap
- [ ] **Q-MBPP**: Basic programming problems in Q.
- [ ] **Custom Metrics**: Flexible framework for adding custom evaluation metrics beyond Pass@k.
- [ ] **Sandboxed Execution**: Secure, isolated code evaluation environment.
- [ ] **MCP Server for Q**: Support MCP/REST API for Q execution to remove the PyKX dependency requirement.
- [ ] **Native Q Execution**: Remove the PyKX dependency.



---

## Troubleshooting

### vLLM Issues

**Problem**: vLLM (NCCL library) fails with NVLS-related errors on single-node setups.

**Solution**: Disable NVLS to force fallback:
```bash
export NCCL_NVLS_ENABLE=0
```

---

## Citation
```bibtex
@software{q_evaluation_harness,
  title={Q Evaluation Harness: Benchmarking LLMs on Q/kdb+ Code Generation},
  author={Miahi, Erfan and Morrison, Andrew},
  year={2025},
  url={https://github.com/kxsystems/q-evaluation-harness}
}
```

---

## License
MIT License - see [LICENSE](LICENSE) for details.
