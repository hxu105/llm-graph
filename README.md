# LLM-Graph

**LLM-Graph** is a research framework dedicated to exploring the intersection of Large Language Models (LLMs) and Graph Machine Learning. This project investigates different paradigms of graph-text integration, specifically focusing on aligning graph modalities within hidden representation spaces and leveraging LLMs for complex graph reasoning tasks.

## 🚀 Overview

Current implementations in this repository are inspired by state-of-the-art methodologies:
* **Modality Alignment:** Based on [LLaGA](https://arxiv.org/abs/2402.08170) and [GraphToken](https://arxiv.org/abs/2402.08170), which aim to map graph structures directly into the LLM's latent space.

---

## 🛠 Installation

### 1. Environment Setup
We recommend using **Python 3.10.0** or higher. You can set up your environment using Conda or a virtual environment:

```bash
# Using Conda (Recommended)
conda create --name llm4graph python=3.10
conda activate llm4graph

# OR using venv
python -m venv llm4graph
source llm4graph/bin/activate
```

### 2. Dependencies
Install the versions of PyTorch and PyG that match your CUDA environment. For CUDA 12.4 and PyTorch v2.6.0, use the following:

```bash
# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric and dependencies
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 3. LLM & Utility Packages

```bash
pip install transformers==4.57.3 datasets==4.4.1 fire
```

## 📊 Dataset & Preprocessing
Currently, the project focuses on **long-range graph datasets**. We use [CityNetwork](https://arxiv.org/abs/2503.09008) as our primary benchmark to evaluate the LLM's ability to capture global graph properties. Future updates will include general graph benchmarks.

### Data Preprocessing
To prepare your data for LLM fine-tuning, you must convert graph structures into Hugging Face dataset cards.

```python
from data import convert_HF

# Define the task-specific instruction for the LLM
prompt = "Proper instructions fit the task"

# Process multiple TAGs 
for tag in ["Cora", "CiteSeer", "PubMed"]:
  convert_HF(root='./data/', name=tag, prompt=prompt)
```
[!TIP] You can perform data augmentation or modify k-hop-subgraph settings by passing additional parameters to convert_HF.

## 🏋️ Training
The current backbone is based on the Llama architecture. Support for additional LLM backbones is currently under development.

To start the fine-tuning process, run:

```bash
bash scripts/run_clm.sh <tag_name>
```

## 🧪 Evaluation
To evaluate the model's performance and generate responses via chat completion, use the provided evaluation script:

```bash
bash scripts/chat_completion.sh
```
