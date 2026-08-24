# LLM-Enhanced Knowledge Graph Quality Evaluation
### Comparative Study on FB15k-237 and WN18RR

This repository contains the source code, datasets, and experimental configurations for evaluating the capability of Large Language Models (LLMs) to validate and refine Knowledge Graphs (KGs). The study focuses on two benchmark datasets: **FB15k-237** (real-world facts) and **WN18RR** (lexical semantics).

## Key Contributions
* **Relation-Aware Corruption Engine:** A custom pipeline to generate semantically plausible "type-safe" negative samples.
* **Multi-Strategy Prompting:** Evaluation across four strategies (S1–S4), including Zero-Shot, Few-Shot, Chain-of-Thought, and **Expert Persona** role-play (S4).
* **Failure Taxonomy:** A classification of LLM errors including Ontological Over-Generalization, Plausibility Traps, Structural Skepticism, and Semantic Ambiguity.
* **Topological Analysis:** Measuring how LLM-driven graph pruning reshapes network density and connectivity to mitigate "embedding pollution."

## Repository Structure
```text
├── data/                   # Raw input (.tsv, .txt) and intermediate (.csv) data
├── scripts/                # Python processing and evaluation logic
│   ├── sampling.py         # Pre-processing and Stratified sampling (N=500)
│   ├── network_analysis.py # Initial topological summary and visualization
│   ├── corruption.py       # Relation-aware triplet corruption (Type-safe)
│   ├── mapping.py          # ID-to-Text natural language mapping
│   ├── llama_inference.py  # Llama-3-70B vLLM inference logic
│   ├── gpt_inference.py    # GPT-5.4-mini API inference logic
│   └── plots.py            # Result visualization (Heatmaps, Radar charts)
├── slurm/                  # Batch script for LARCC HPC execution
│   └── llama_inf.slurm     # GPU cluster submission script
├── results/                # Output logs, CSV results, and final plots
├── run_gpt.sh              # Local execution wrapper (macOS/Linux)
├── run_gpt.bat             # Local execution wrapper (Windows)
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation
```

## Setup & Installation
1. Clone the repository:
   
   git clone https://github.com/alizainkaimkhan/llm-based-knowledge-graph-eval.git
   cd llm-kg-evaluation-project
   
2. Install dependencies:
   
   pip install -r requirements.txt
   

## How to Run

### 1. Data Preparation Pipeline
Run these scripts from the project root to generate the evaluation set:

python scripts/sampling.py
python scripts/corruption.py
python scripts/mapping.py


### 2. LLM Inference
#### **Remote Execution (LARCC Cluster)**
Submit the job to the GPU partition for Llama-3-70B inference:

sbatch slurm/llama_inf.slurm


#### **Local Execution (macOS / Linux / Windows)**
Use the wrappers for GPT-5.4 mini inference. **Note:** Ensure your `OPENAI_API_KEY` is set inside the script.
* **macOS/Linux:** chmod +x run_gpt.sh && ./run_gpt.sh
* **Windows:** Double-click run_gpt.bat

### 3. Analysis & Visualization
Generate final F1-score heatmaps, radar charts and confusion matrices:

python scripts/plots.py


## Future Work
* **Supervised Fine-Tuning (SFT):** Training models specifically for graph-judgment tasks.
* **Agentic Multi-Model Consensus:** Collaborative auditing between Llama and GPT.
* **Knowledge Graph of Thoughts (KGoT):** Externalizing LLM reasoning into dynamic graph structures.
* **Interactive Refinement:** Human-in-the-loop safeguards and semantic filtering.

## Contributors & Affiliation
* **Ali Zain Kaim Khan**
* **Rafia Fayyaz**
* **Zeeshan Akram**
* **Course:** CSE 694: Deep Learning on Graphs - Spring 2026
