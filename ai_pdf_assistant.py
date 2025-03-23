import os
import ollama
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.docstore.document import Document

# Path to your PDF folder
PDF_FOLDER = os.path.expanduser("~/ai_docs")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with page numbers."""
    text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(Document(page_content=page_text, metadata={"page": i+1, "source": pdf_path}))
    return text

def load_pdfs():
    """Loads and combines text from all PDFs in the folder into structured documents."""
    all_documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            all_documents.extend(extract_text_from_pdf(pdf_path))
    return all_documents

def find_relevant_chunks(question, documents):
    """Finds the most relevant text chunks based on the user's question."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Retrieve top 2 most relevant chunks
    similar_docs = vector_store.similarity_search(question, k=1)
    return [f"(Page {doc.metadata['page']}): {doc.page_content}" for doc in similar_docs]

def chat_with_pdf(question):
    """Answers a question based on the most relevant document sections using Ollama."""
    documents = load_pdfs()

    if not documents:
        print("\n🚨 No text extracted from PDFs. Ensure they are not scanned images.")
        return

    # Split text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    text_chunks = splitter.split_documents(documents)

    # Find relevant sections
    selected_chunks = find_relevant_chunks(question, text_chunks)

    if not selected_chunks:
        print("\n⚠️ No relevant sections found for the question.")
        return

    # Prepare the prompt with relevant document sections
    selected_text = "\n".join(selected_chunks)

    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "You are an AI expert answering based on provided documents."},
            {"role": "user", "content": f"Based on these documents, answer: {question}\n\n{selected_text}"}
        ],
        options={"num_predict": 256}  # Limits response length for speed
    )

    print("\n💡 AI Response:\n", response.get("message", {}).get("content", "No response."))

if __name__ == "__main__":
    print("📄 AI PDF Assistant (Dynamic Chunk Selection) is ready!")
    
    while True:
        user_input = input("\nAsk a question (or type 'exit' to quit): ").strip()
    
        if not user_input:  # Prevent processing on empty input
            print("⚠️ No question entered. Please try again.")
            continue
    
        if user_input.lower() == "exit":
            break
    
        chat_with_pdf(user_input)

