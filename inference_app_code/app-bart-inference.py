import gradio as gr
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import emoji
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths relative to the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH = os.path.join(CURRENT_DIR, 'Bart_base_emoji/bart_base_full_finetune_emoji_save')
SAVED_TOK_PATH = os.path.join(CURRENT_DIR, 'Bart_base_emoji/tokenizer-emoji')

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
    
    return summary

# Load models and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_PATH).to(device)
tokenizer = BartTokenizer.from_pretrained(SAVED_TOK_PATH)

# Define Gradio interface function
def summarize_for_gradio(text):
    """
    Summarizes the given text using the provided tokenizer and model.

    Args:
        text (str): The dialogue text to be summarized.

    Returns:
        str: The summarized dialogue.
    """
    summary = summarize(tokenizer, model, text)
    return summary[0]

iface = gr.Interface(
    fn=summarize_for_gradio,
    inputs=gr.Textbox(lines=10, placeholder="Enter dialogue here...", label="Input Dialogue"),
    title="Bart_base fine tuned on SAMsum dataset",
    outputs=gr.Textbox(label="Generated Summary")
)

iface.launch(share=True)
