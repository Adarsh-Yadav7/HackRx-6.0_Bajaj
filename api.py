
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import uvicorn

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Path to your local FAISS vector store
VECTOR_DB_PATH = "faiss_index"

# Prompt template for the insurance claim decision
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

# Create PromptTemplate instance
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Input data model
class QueryInput(BaseModel):
    query: str

# POST endpoint to process query
@app.post("/query")
async def get_claim_decision(data: QueryInput):
    try:
        # Load vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(data.query, k=4)

        # Load LLM (LLaMA3 on Groq)
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # Load QA chain
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        # Run chain on input
        response = chain.run({
            "input_documents": docs,
            "question": data.query
        })

        # Return parsed JSON
        try:
            return json.loads(response)
        except Exception:
            return {"error": "Could not parse response", "raw_response": response}

    except Exception as e:
        return {"error": str(e)}

# Run server
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
