from huggingface_hub import login
login(token=HF_TOKEN) # Replace HF_TOKEN with your actual Hugging Face token

import streamlit as st
import os
import torch
import random
import numpy as np
import pandas as pd
import transformers

from glob import glob
from tqdm.notebook import tqdm
import warnings
warnings.simplefilter('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

set_seed(42)

if 'embedding_model' not in st.session_state:
    from sentence_transformers import SentenceTransformer
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = st.session_state.embedding_model

df = pd.read_csv(YOUR_CSV_FILE_PATH)  # Replace with your actual CSV file path
df['text'] = df['text'] + ' ' + df['labels']
df = df[['text']]

import PyPDF2

if 'llm_pipeline' not in st.session_state:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    model_name = "google/gemma-2-9b-it" # Change to other LLMs (e.g. "google/gemma-2-27b-it", "Qwen/Qwen2.5-7B-Instruct") as needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    st.session_state.tokenizer = tokenizer
    st.session_state.llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0,
                                             #  , do_sample=False  Add this line when an error occurs (e.g. using Qwen2.5-7B-Instruct)
                                             )

def extract_text_from_pdf(pdf_path, max_tokens=150):
    text = ""
    chunks = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    
    tokens = tokenizer(text, return_tensors="np", truncation=False)["input_ids"][0]
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks

tokenizer = st.session_state.tokenizer
pdf_paths = glob(YOUR_PDF_FILE_PATHS)  # Replace with your actual PDF file paths
pdf_texts = []
for pdf_path in pdf_paths:
    pdf_texts.extend(extract_text_from_pdf(pdf_path, max_tokens=150))

if 'csv_embeddings' not in st.session_state:
    st.session_state.csv_embeddings = embedding_model.encode(df['text'].values, show_progress_bar=True)
if 'pdf_embeddings' not in st.session_state:
    st.session_state.pdf_embeddings = embedding_model.encode(pdf_texts, show_progress_bar=True)

csv_embeddings = st.session_state.csv_embeddings
pdf_embeddings = st.session_state.pdf_embeddings

import faiss

csv_index = faiss.IndexFlatL2(csv_embeddings.shape[1])
csv_index.add(csv_embeddings.astype('float32'))

pdf_index = faiss.IndexFlatL2(pdf_embeddings.shape[1])
pdf_index.add(pdf_embeddings.astype('float32'))

text_gen_pipeline = st.session_state.llm_pipeline


def summarize_text(text, pipeline, max_new_tokens=512):
    prompt = (
    "YOUR_SUMMARY_PROMPT:\n\n" # This should be defined in your context
    f"{text}\n\nSummary:")
    outputs = pipeline(prompt, max_new_tokens=max_new_tokens)
    generated_text = outputs[0]['generated_text'].strip()

    if "Summary:" in generated_text:
        summary = generated_text.split("Summary:")[-1].strip()
    else:
        summary = generated_text

    return f"\n{summary}\n"

def ask_question(question, embedding_model, csv_index, pdf_index, csv_data, pdf_data, text_gen_pipeline, max_new_tokens=512, k=10):
    question_embedding = embedding_model.encode([question], convert_to_tensor=False)
    
    csv_distances, csv_indices = csv_index.search(np.array(question_embedding), k=k)
    pdf_distances, pdf_indices = pdf_index.search(np.array(question_embedding), k=k)
    
    csv_texts = []
    pdf_texts = []
    
    for idx in csv_indices[0]:
        csv_text = csv_data.iloc[idx]['text']
        csv_texts.append(csv_text)
        print(f"CSV Hit:\n{csv_text}\n")
    
    for idx in pdf_indices[0]:
        pdf_text = pdf_data[idx]
        pdf_texts.append(pdf_text)
        print(f"PDF Hit:\n{pdf_text}\n")
    
    retrieved_texts = []
    
    if csv_texts:
        combined_csv_text = "\n".join(csv_texts)
        csv_summary = summarize_text(combined_csv_text, text_gen_pipeline, max_new_tokens=512) #256
        print(f"CSV Summary:\n{csv_summary}\n")
        retrieved_texts.insert(0, f"From experimental data: {csv_summary}")
    
    if pdf_texts:
        combined_pdf_text = "\n".join(pdf_texts)
        pdf_summary = summarize_text(combined_pdf_text, text_gen_pipeline, max_new_tokens=512) #256
        print(f"PDF Summary:\n{pdf_summary}\n")
        retrieved_texts.append(f"From paper findings: {pdf_summary}")
    
    retrieved_context = "\n".join(retrieved_texts)
    print(f"Related Information:\n{retrieved_context}\n")
    input_text = f"Question: {question}\n\nRelated Information: \n\n{retrieved_context}\n\nAnswer: "
    
    outputs = text_gen_pipeline(input_text, max_new_tokens=max_new_tokens)
    assistant_response = outputs[0]['generated_text'].strip()

    return assistant_response, csv_texts, pdf_texts

st.title("Material Dual-Source Knowledge RAG (MDSK-RAG) Chatbot")
st.markdown("Enter your question below to get an answer.")
question = st.text_input("Enter your question:")
if st.button("Get related information and answer"):
    if question:
        with st.spinner("Generating ..."):
            answer, csv_texts, pdf_texts = ask_question(question, embedding_model, csv_index, pdf_index, df, pdf_texts, text_gen_pipeline)
        st.write(answer)
        st.write("### CSV Texts:")
        for text in csv_texts:
            st.write(text)
        st.write("### PDF Texts:")
        for text in pdf_texts:
            st.write(text)