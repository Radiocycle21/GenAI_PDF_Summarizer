import os
import streamlit as st

st.set_page_config(page_title="GenAI PDF Summarizer", layout="wide")
#st.markdown("✅ App is running!")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import PyPDF2
from transformers import pipeline
import textwrap

# Load Model
@st.cache_resource()
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Extract Text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])

# Summarize Text
def summarize_text(text):
    chunks = textwrap.wrap(text, width=512)  # Keep chunk size small
    summary = []
    for chunk in chunks:
        input_length = len(chunk.split())  # Get the number of words in input
        max_length = max(30, int(input_length * 0.5))  # Adjust max_length dynamically
        try:
            result = summarizer(chunk, max_length=max_length, min_length=10, do_sample=False)
            summary.append(result[0]['summary_text'])
        except Exception as e:
            summary.append(f"[Error summarizing: {str(e)}]")
    return " ".join(summary)

# Streamlit UI
st.title("📄 PDF & Text Summarizer")

uploaded_file = st.file_uploader("📂 Upload PDF (Max 2MB)", type="pdf")
user_text = st.text_area("✍️ Paste Text (Max 1000 Words)", height=200)
summary_result = None

if st.button("🔍 Summarize"):
    if uploaded_file:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("❌ File size exceeds 2MB limit.")
        else:
            with st.spinner("Processing..."):
                summary_result = summarize_text(extract_text_from_pdf(uploaded_file))
    elif user_text.strip():
        if len(user_text.split()) > 500:
            st.error("❌ Text exceeds 500-word limit!")
        else:
            with st.spinner("Summarizing..."):
                summary_result = summarize_text(user_text)
    else:
        st.warning("⚠️ Please upload a PDF or enter text.")

# Display & Download Summary
if summary_result:
    st.subheader("📄 Summary:")
    st.write(summary_result)
    st.download_button("📥 Download Summary", summary_result.encode('utf-8'), "summary.txt", "text/plain")
