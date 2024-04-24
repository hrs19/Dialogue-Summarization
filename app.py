import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import emoji
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:',device)

# Define paths relative to the current script's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH1 = os.path.join(CURRENT_DIR, 'models/BART_BASE/bart_base_full_finetune_emoji_save-20')
SAVED_TOK_PATH1 = os.path.join(CURRENT_DIR, 'models/BART_BASE/bart_base_full_finetune_model_tokenizer')


SAVED_MODEL_PATH2 = os.path.join(CURRENT_DIR, 'models/FLAN_T5_BASE/flan_t5_base_lora_finetune_emoji_save_adapter')  # Change this to your second model path
SAVED_TOK_PATH2 = os.path.join(CURRENT_DIR, 'models/FLAN_T5_BASE/tokenizer-emoji_t5')

# SAVED_TOK_PATH = 'tokenizer-emoji_t5'
SAVED_MODEL_TOK = AutoTokenizer.from_pretrained(SAVED_TOK_PATH2)#.to(device)
from peft import PeftModel, PeftConfig
# Load peft config for pre-trained checkpoint etc.
# peft_model_id = "./flan_t5_base_lora_finetune_emoji_save_adapter"
config = PeftConfig.from_pretrained(SAVED_MODEL_PATH2)#.to(device)
combined_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path).to(device)
combined_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
combined_model = PeftModel.from_pretrained(combined_model, SAVED_MODEL_PATH2).to(device)
combined_model.resize_token_embeddings(len(SAVED_MODEL_TOK))




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

# Load models and tokenizers
model1 = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_PATH1).to(device)
tokenizer1 = AutoTokenizer.from_pretrained(SAVED_TOK_PATH1)
# model2 = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_PATH2).to(device)
# tokenizer2 = BartTokenizer.from_pretrained(SAVED_TOK_PATH2)

# BACKGROUND_IMAGE_PATH = os.path.join(CURRENT_DIR, '/background.jpg')
# # Custom CSS for background image and styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-image: url('data:image/jpg;base64,{st.file_uploader(BACKGROUND_IMAGE_PATH, type=["jpg", "jpeg", "png"]).getvalue().decode("utf-8")}');
#         background-size: cover;    }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the background image
BACKGROUND_IMAGE_PATH = os.path.join(CURRENT_DIR, 'assets','background.jpg')
bg_image_base64 = get_base64_of_bin_file(BACKGROUND_IMAGE_PATH)
st.set_page_config(page_title="Summarizer", page_icon="", layout="wide", )     

# Custom CSS for background image and styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp p {{
    background: rgba(255, 255, 255, 0.8);
        # color: #FF0000; /* red color for better contrast */
        # text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* shadow effect */
        text-align: center;
    }}    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.title("Meeting Summarizer")
st.write("Enter the dialogue/meeting transcripts that you want to summarize.")

# Layout
col1, col2 = st.columns(2)

with col1:
    text = st.text_area("Input Dialogue", height=500)

with col2:
    if st.button("Summarize"):
        if text:
            summary1 = summarize(tokenizer1, model1, text)[0]
            summary2 = summarize(SAVED_MODEL_TOK,combined_model, text)[0]
            
            st.subheader("BART_BASE Full Finetuned Model Summary:")
            st.text_area("Summary 1", summary1, height=100)
            
            st.subheader("FLAN_T5_BASE LoRA Finetuned Model Summary:")
            st.text_area("Summary 2", summary2, height=100)
        else:
            st.warning("Please enter some dialogue text to summarize.")
