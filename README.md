# Dialogue Summarization with Language Models

This repository contains a series of Jupyter notebooks that demonstrate the process of fine-tuning and evaluating various language models on the dialogue summarization task using the SAMSum dataset. The focus is on fine-tuning BART and FLAN-T5 models to effectively capture the nuances and context of human-like dialogue.

## Overview

The project aims to explore different fine-tuning strategies to enhance the ability of language models to generate accurate and context-rich summaries of dialogues. This involves understanding inter-speaker relationships, detecting thematic shifts within dialogues, and attributing statements correctly within the generated summaries.

## Repository Structure

Below is a description of the key notebooks in this repository:

- `experiment-notebooks/BART-base/bart_base_encoder.ipynb`:
    - Training the bart base model by finetuning only the encoder layers

## Getting Started

To run these notebooks, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hrs19/Dialogue-Summarization.git
   cd Dialogue-Summarization
