import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
import openai
# from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from pathlib import Path
import shutil
from io import BytesIO

# Set OpenAI API Key
#openai.api_key = st.secrets["openai_key"]
Groq.api_key = st.secrets["groq_key"]

def main():
    st.title("DocTalk, talk to your docs  - Developed by Abhyas Manne")
    st.write("Upload one or more PDF files")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF files uploaded.")
        merged_pdf_path = merge_pdfs(uploaded_files)
        if merged_pdf_path:
            st.write("PDF files merged successfully!")
            try:
                with open(merged_pdf_path, "rb") as file:
                    st.download_button(
                        label="Download Merged PDF",
                        data=file,
                        file_name="merged_document.pdf",
                        mime="application/pdf"
                    )
                index = None
                storage_dir = None
                if "temp_dir" not in st.session_state:
                    st.session_state.temp_dir = tempfile.mkdtemp()
                if "index" not in st.session_state:  # Initialize the index only once
                    index, storage_dir = index_pdf(merged_pdf_path,st.session_state.temp_dir)
                    if index is None or storage_dir is None:
                        st.error("Failed to index PDF. Please try again.")
                    else:
                        st.session_state.index = index
                        st.session_state.storage_dir = storage_dir
                        st.write("Using the existing index..")



       #     index, storage_dir = index_pdf(merged_pdf_path)

       #         if "index" not in st.session_state:  # Initialize the index only once
       #             st.session_state.index, st.session_state.storage_dir = index_pdf(merged_pdf_path)

                if st.session_state.index:
                    st.write("PDF indexed successfully! You can now ask questions. Please wait a few seconds..")
                   
                    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
                        st.session_state.messages = [
                            {"role": "assistant", "content": "Welcome to DocTalk"}
                        ]
                    
                    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
                            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
                    summary = st.session_state.chat_engine.chat("Summarize briefly")
                    st.write("Brief summary of the uploaded documents:")
                    st.write(summary.response)
                    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    for message in st.session_state.messages: # Display the prior chat messages
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    
                    # If last message is not from assistant, generate a new response
                    if st.session_state.messages[-1]["role"] != "assistant":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = st.session_state.chat_engine.chat(prompt)
                                st.write(response.response)
                                message = {"role": "assistant", "content": response.response}
                                st.session_state.messages.append(message) # Add response to message history

                # Clean up temporary files
                if st.session_state.storage_dir and os.path.exists(st.session_state.storage_dir):
                    shutil.rmtree(st.session_state.storage_dir)

            finally:
                os.remove(merged_pdf_path)

def merge_pdfs(files):
    pdf_writer = PdfWriter()
    temp_merged_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        for uploaded_file in files:
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            for page_num in range(len(reader.pages)):
                pdf_writer.add_page(reader.pages[page_num])
        pdf_writer.write(temp_merged_pdf)
        temp_merged_pdf.close()
        return temp_merged_pdf.name
    except Exception as e:
        st.error(f"An error occurred while merging PDFs: {e}")
        return None
    finally:
        temp_merged_pdf.close()
#@st.cache_resource(show_spinner=False)
def index_pdf(pdf_path,temp_dir):
    try:
        storage_dir = Path(temp_dir) / "storage"
        pdf_dir = storage_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_path, pdf_dir / "merged_document.pdf")

        with st.spinner("Indexing documents..."):
            docs = SimpleDirectoryReader(pdf_dir).load_data()
            service_context = ServiceContext.from_defaults(llm=Groq(model="llama3-70b-8192", api_key=Groq.api_key, temperature=0.2, system_prompt="You are assistant researcher who is a famous researcher to evaluate scientific articles. It is extremely important research. Before responding verify the context very carefully. Your response should be very clear and specific, wherever possible quote references from the context.  If the questioned cannot be answered with the information within the context provided, then reply that you could not find the relavent information in the context, do not hallucinate. Be very helpful" ))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            index.set_index_id("pdf_index")
            index.storage_context.persist(storage_dir)

        return index, storage_dir
    except Exception as e:
        st.error(f"An error occurred while indexing PDF: {e}")
        return None, None

if __name__ == "__main__":
    main()
