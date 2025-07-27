from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from search_system import PDFSearchSystem
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
search_system = PDFSearchSystem(use_azure_llm=True)

class SearchQuery(BaseModel):
    query: str

class IndexSources(BaseModel):
    sources: list[str]

@app.post("/create_index")
def create_index(sources: IndexSources):
    success = search_system.create_index(sources.sources)
    if success:
        search_system.load_index()
        return {"message": "Index created successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to create index.")

@app.post("/search")
def search(query: SearchQuery):
    if not search_system.vectorstore:
        raise HTTPException(status_code=400, detail="Vectorstore not loaded. Please create an index first.")
    results = search_system.search(query.query)
    return {"results": results}

@app.post("/qa")
def qa(query: SearchQuery):
    if not search_system.qa_chain:
        if search_system.vectorstore:
            success = search_system.setup_qa_chain()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to set up QA chain.")
        else:
            raise HTTPException(status_code=400, detail="Vectorstore not loaded. Please create an index first.")

    answer, source_docs = search_system.ask_question(query.query)
    return {"answer": answer, "source_documents": source_docs}

@app.on_event("startup")
def startup_event():
    search_system.load_index()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)