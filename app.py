import streamlit as st
from rag import load_pdf, retriever, rag_chain
import tempfile
import os


# Create the Streamlit app
st.title("RAG - Query the PDF")

# Upload the file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Define a directory within the project to store temporary files
    temp_dir = os.path.join(os.getcwd(), "temp_files")
    
    # Create the directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define the full path for the uploaded file
    path = os.path.join(temp_dir, uploaded_file.name)

    with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

    
    # Load the PDF file
    with st.spinner("Loading PDF..."):
        documents = load_pdf(path)

    # Initialize the retriever
    retrieve = retriever(documents)

    # Initialize the RAG chain
    rag = rag_chain(retrieve)

    # Input for the user's question
    question = st.text_input("Ask your question:")

    if question:
        # Get the response from RAG
        with st.spinner("Getting response..."):
            response = rag.invoke(question)

        # Display the response
        st.write("Bot Response: ", response)
else:
    st.info("Please upload a PDF file to proceed.")
