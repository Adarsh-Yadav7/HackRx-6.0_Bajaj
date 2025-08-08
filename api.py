import sys
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import uvicorn

# Configure environment for Render deployment
sys.path.append('/opt/render/.local/lib/python3.9/site-packages')
os.environ['NO_CUDA'] = '1'  # Disable CUDA dependencies
os.environ['FAISS_NO_AVX2'] = '1'  # Disable AVX2 instructions
os.environ['FAISS_OPT_LEVEL'] = 'generic

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure paths
VECTOR_DB_PATH = "faiss_index"

# Define the prompt template
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

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Input data model
class QueryInput(BaseModel):
    query: str

@app.post("/query")
async def process_claim(data: QueryInput):
    try:
        # Load vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(data.query, k=4)

        # Initialize LLM
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # Setup QA chain
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        # Process query
        response = chain.run({
            "input_documents": docs,
            "question": data.query
        })

        # Parse and return response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error": "Invalid response format",
                "raw_response": response
            }

    except Exception as e:
        return {"error": str(e)}

# Production-ready server configuration
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info"
    )
