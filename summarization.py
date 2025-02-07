
import streamlit as st
import PyPDF2
from transformers import pipeline
import textwrap

# Load Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extract Text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])

# Summarize Text
def summarize_text(text):
    chunks = textwrap.wrap(text, width=1024)
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

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
        if len(user_text.split()) > 1000:
            st.error("❌ Text exceeds 1000-word limit!")
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
