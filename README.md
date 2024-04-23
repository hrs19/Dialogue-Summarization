# Dialogue Summarization with Language Models

This repository contains a series of Jupyter notebooks that demonstrate the process of fine-tuning and evaluating various language models on the dialogue summarization task using the SAMSum dataset. The focus is on fine-tuning BART and FLAN-T5 models to effectively capture the nuances and context of human-like dialogue.

## Overview

The project aims to explore different fine-tuning strategies to enhance the ability of language models to generate accurate and context-rich summaries of dialogues. This involves understanding inter-speaker relationships, detecting thematic shifts within dialogues, and attributing statements correctly within the generated summaries.

## Repository Structure

The notebooks are contained within the `experiment-notebooks` directory. Each sub-directory corresponds to a specific model and contains notebooks for different fine-tuning approaches:

- `experiment-notebooks/BART-base`
  - `bart_base_encoder.ipynb`: Encoder-only fine-tuning strategy for BART-base.
  - `bart_base_full_finetune.ipynb`: Full model fine-tuning for BART-base.

- `experiment-notebooks/BART-large`
  - `bart_large_encoder_only_finetune.ipynb`: Encoder-only fine-tuning for BART-large.
  - `bart_large_selective_layers.ipynb`: Selective layers fine-tuning for BART-large.
  - `bart-large-cnn-lora.ipynb`: LoRA fine-tuning for BART-large-cnn.

- `experiment-notebooks/FLAN-T5-base`
  - `flan-t5-base-lora-finetune.ipynb`: LoRA fine-tuning for FLAN-T5-base model.

- `experiment-notebooks/FLAN-T5-small`
  - `flan-t5-small-full_finetune.ipynb`: Full fine-tuning for FLAN-T5-small model.

## Getting Started

To run these notebooks, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hrs19/Dialogue-Summarization.git
   cd Dialogue-Summarization

2. **Install Dependencies**: Ensure that Jupyter Notebook or JupyterLab is installed on your system. Additionally, install all required Python packages as listed in `requirements.txt`.

    ```
    pip install -r requirements.txt
    ```

3. **Running the Notebooks**: Navigate to the `experiment-notebooks` directory, open the sub-directory for the model you wish to fine-tune, and start the Jupyter Notebook.

    ```
    jupyter notebook <notebook_name>.ipynb
    ```

    Or for JupyterLab users:

    ```
    jupyter lab <notebook_name>.ipynb
    ```
