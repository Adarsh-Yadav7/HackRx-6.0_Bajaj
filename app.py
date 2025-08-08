
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

PDF_FOLDER = r"C:\Users\vishw\OneDrive\Desktop\Gen_AI\Bajaj\documents"
VECTOR_DB_FOLDER = "faiss_index"

# Load PDFs
def load_documents(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# Split into chunks
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Create & save vectorstore
def create_and_save_vectorstore(chunks, path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(path)
    return vectorstore

# Load saved vectorstore
def load_vectorstore(path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# Prompt Template (strict)
prompt_template = """
You are an expert insurance claim assistant.

You will receive:
- A user query (natural language with age, city, surgery, policy time).
- Policy clauses as context.

Your task:
- Extract age, procedure, city, duration.
- Match it with the context clauses.
- Decide if claim is APPROVED or REJECTED.
- If amount is mentioned, give it. Else, return "N/A".
- Give short justification with clause reference.

⚠️ Return ONLY this JSON — do NOT add explanation or text before/after:

{{
  "decision": "approved/rejected",
  "amount": "amount in INR or N/A",
  "justification": "reason with clause reference"
}}

Context: {context}
Query: {question}
"""

# Run logic once to prepare vectorstore
def prepare_vectorstore():
    if os.path.exists(os.path.join(VECTOR_DB_FOLDER, "index.faiss")):
        print("✅ Vectorstore already exists. Loading...")
        return load_vectorstore(VECTOR_DB_FOLDER)
    else:
        print("⏳ Creating vectorstore...")
        documents = load_documents(PDF_FOLDER)
        chunks = chunk_documents(documents)
        return create_and_save_vectorstore(chunks, VECTOR_DB_FOLDER)

# Get LLM response
def get_decision(query):
    vectorstore = load_vectorstore(VECTOR_DB_FOLDER)
    docs = vectorstore.similarity_search(query, k=4)

    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    response = chain.run({"input_documents": docs, "question": query})
    return response  # Raw string (json.loads to be done in FastAPI)

# Optional manual test
if __name__ == "__main__":
    prepare_vectorstore()
    test_query = "46-year-old male, knee surgery in Pune, 3-month-old policy"
    print(get_decision(test_query))
