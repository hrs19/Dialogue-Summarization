import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import emoji
from peft import PeftModel, PeftConfig
import os

# Checking for available device (CPU or CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths relative to the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_TOK_PATH = os.path.join(CURRENT_DIR, 'tokenizer-emoji_t5')
PEFT_MODEL_PATH = os.path.join(CURRENT_DIR, 'flan_t5_base_lora_finetune_emoji_save_adapter')

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(SAVED_TOK_PATH)

# Configuring and loading the PEFT model
config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path).to(device)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH).to(device)
model.resize_token_embeddings(len(tokenizer))

def summarize(tokenizer, model, text):
    """
    Summarizes the given text using the provided tokenizer and model.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the input text.
        model (PeftModel): The model used for summarization.
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    # Convert emojis to text representation
    text = emoji.demojize(text, delimiters=('<', '>'))
    
    # Tokenize the input text and prepare inputs for model
    inputs = tokenizer(f"Summarize dialogue >>\n {text}", return_tensors="pt", max_length=1000, truncation=True, padding="max_length").to(device)
    
    # Generate summary output with max 100 token limit from the model using beam search with 4 beams
    summary_ids = model.generate(inputs=inputs.input_ids, num_beams=4, max_length=100, early_stopping=True)
    
    # Decode the generated token ids to human-readable text
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    
    return summary[0]

def summarize_dialogue(text):
    """
    Wrapper function for Gradio interface, to summarize dialogue text.

    Args:
        text (str): The dialogue text to be summarized.

    Returns:
        str: The summarized dialogue.
    """
    return summarize(tokenizer, model, text)

# Create a Gradio interface
iface = gr.Interface(
    fn=summarize_dialogue,
    inputs=gr.Textbox(lines=10, placeholder="Enter dialogue here...", label="Input Dialogue"),
    title="Flan t5 fine tuned using LoRA on SAMsum dataset",
    outputs=gr.Textbox(label="Generated Summary")
)

# Launch the interface
iface.launch(share=True)
