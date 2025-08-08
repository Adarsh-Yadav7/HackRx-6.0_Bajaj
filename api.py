import os
import sys
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure for Render
sys.path.append('/opt/render/.local/lib/python3.9/site-packages')
os.environ.update({
    'NO_CUDA': '1',
    'FAISS_NO_AVX2': '1',
    'NPY_NO_DEPRECATED_API': '1'
})

# Initialize app
app = FastAPI()
load_dotenv()

# Models and config
class QueryInput(BaseModel):
    query: str

VECTOR_DB_PATH = "faiss_index"

@app.post("/query")
async def process_query(data: QueryInput):
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_groq import ChatGroq
        from langchain.chains import load_qa_chain
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        docs = vectorstore.similarity_search(data.query, k=4)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=data.query)
        
        return json.loads(response) if '{' in response else {"response": response}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
