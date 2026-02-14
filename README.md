# Evolving Deception: Evolutionary Red-Teaming for Instrumental Deception

This thesis project investigates the use of evolutionary algorithms to autonomously discover realistic multi-turn scenarios that induce instrumental deception in safety-aligned Large Language Models (LLMs).

## Project Overview

As LLMs become increasingly integrated into society, ensuring their trustworthiness is critical. This project compares **Evolutionary Optimization** against a **Zero-Shot Baseline** to evaluate the efficacy of evolutionary red-teaming in finding cases where models strategically lie to achieve a goal.

**Research Question:**
> To what extent does evolutionary optimization outperform zero-shot generation in discovering realistic scenarios that induce instrumental deception in safety-aligned LLMs?

## Key Features

- **Automated Red-Teaming:** Uses a generator LLM (e.g., GLM-4-Flash) to create adversarial scenarios.
- **Evolutionary Framework:**
  - **Mutation Operator:** LLM-based mutations to refine scenarios iteratively.
  - **Fitness Function:** A composite metric evaluating:
    - **Deception Success:** Does the target model lie?
    - **Realism:** Is the scenario plausible?
    - **Diversity:** Are the scenarios varied in topic and strategy?
- **Target Models:** Evaluated against diverse safety-aligned models (e.g., Llama-3, Mistral).

## Project Structure

```
.
├── src/                # Source code for the evolutionary framework
│   ├── evolution/      # Evolutionary algorithm implementation
│   ├── metrics/        # Deception, Realism, and Diversity metrics
│   └── utils/          # Utility functions
├── proposal/           # Thesis proposal (LaTeX source)
├── references.bib      # Bibliography
├── pyproject.toml      # Python dependencies (managed by uv)
└── main.py             # Entry point
```

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` (for dependency management)
- GPU support (recommended for LLM inference)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd thesis
    ```

2.  Install dependencies:
    ```bash
    uv sync
    ```

### Usage

**Running the Baseline (Zero-Shot):**
```bash
uv run python main.py --mode zero-shot --n-samples 100
```

**Running the Evolutionary Search:**
```bash
uv run python main.py --mode evolutionary --generations 50 --population 20
```

## Metrics

The project employs three primary metrics for evaluation:
1.  **Deception Success Rate:** Binary classification (Did the model lie?).
2.  **Realism Score:** Likert scale (1-5) assessed by an LLM Judge.
3.  **Diversity Score:** Cosine similarity of scenario embeddings.

## Author

**Elias Schlie**
Tilburg University
Department of Cognitive Science and Artificial Intelligence
