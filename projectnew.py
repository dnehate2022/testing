import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

# Function to extract output from response JSON
def extract_output(response_dict):
    return response_dict.get('output', '')

# Your existing code to interact with the conversational retrieval agent
OPENAI_API_KEY = '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'

# Instantiate the OpenAIEmbeddings class with your API key
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Function to load and split PDF document
def load_and_split_pdf(pdf_bytes):
    # Create a temporary file to store the PDF bytes
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_pdf_file.write(pdf_bytes)
    temp_pdf_file.close()

    # Use the path of the temporary file to load and split the PDF document
    pdf_loader = PyPDFLoader(temp_pdf_file.name)
    pdf_pages = pdf_loader.load_and_split()

    # Process each page and extract text
    texts = []
    for page_content in pdf_pages:
        texts.append(page_content)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_texts = text_splitter.split_documents(texts)

    # Delete the temporary file
    temp_pdf_file.close()
    return chunked_texts

# Function to create vector store and retriever
def create_vector_store_and_retriever(chunked_texts):
    embeddings = openai_embeddings
    db = FAISS.from_documents(chunked_texts, embeddings)
    retriever = db.as_retriever()
    return retriever

# Function to create tool for retrieval
def create_retrieval_tool(retriever):
    tool = create_retriever_tool(
        retriever,
        "My_Favourite_Topic_ComputerandGenerations",
        "Search and return exact word to word paragraph which in pdf documents regarding Topic ComputerandGenerations with provided pdf text.",
    )
    return tool

# Function to create conversational retrieval agent
def create_conversational_retrieval_agent_toolkit(llm, tools):
    agent_executor = create_conversational_retrieval_agent(llm, tools)
    return agent_executor

# Streamlit UI
st.title('PDF Insight Finder: Discover Knowledge Within')

# Allow user to upload PDF file
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file is not None:
    # Load and split PDF document
    pdf_bytes = uploaded_file.getvalue()
    chunked_texts = load_and_split_pdf(pdf_bytes)

    # Create vector store and retriever
    retriever = create_vector_store_and_retriever(chunked_texts)

    # Create tool for retrieval
    tool = create_retrieval_tool(retriever)

    # Create conversational retrieval agent
    llm = ChatOpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    agent_executor = create_conversational_retrieval_agent_toolkit(llm, [tool])

    user_question = st.text_input('Ask your question here:')
    
    if st.button('Submit'):
        if user_question:
            # Invoke the agent to get a response
            response = agent_executor.invoke(user_question)
            output_text = extract_output(response)
            
            # Displaying the response with proper alignment and styling
            st.markdown("---")
            st.markdown(f"**Question:** {user_question}")
            st.markdown(f"**Response:** {output_text}")
            st.markdown("---")
        else:
            st.warning('Please enter a question.')
