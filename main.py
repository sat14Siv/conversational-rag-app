from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import os

from pydantic_models import DeleteFileRequest, QueryInput, QueryResponse

from db_utils import insert_file_to_document_store, delete_file_from_document_store, get_all_documents_in_store
from chroma_utils import index_document_to_chroma

from langchain_utils import get_chat_history, get_rag_chain

import logging
import shutil


# Initialize Logging
logging.basicConfig(filename='app.log', level=logging.INFO)


# Initialize FastAPI app
app = FastAPI()



# Upload document (to both document store and vector db)
@app.post('/upload-document')
def upload_and_index_document(file: UploadFile=File(...)):

    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, 
                            detail=f"File Type not supported. Allowed types are {','.join(allowed_extensions)}")

    try:
        # Add file to document store - just storing the filename and creating an ID
        file_id = insert_file_to_document_store(file.filename)

        # Store the doc temporarily on server filesystem
        temp_filepath = 'temp_' + file.filename
        with open(temp_filepath, 'wb'):
            shutil.copy(file.file, temp_filepath)

        # Add to vector database
        success = index_document_to_chroma(temp_filepath, file_id)

        if success:
            pass
        else:
            delete_file_from_document_store(file_id)
        
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    return {}



# Delete document (from both document and vector db)
@app.post('/delete-document')
def delete_document(request: DeleteFileRequest):
    # Delete the file from document store
    db_delete_success = delete_file_from_document_store(request.file_id)

    # Delete all the chunks, embeddings from the vector store
    chroma_delete_success = delete_file_from_chroma_db(request.file_id)

    return {}



# List all uploaded documents
@app.post('/list-documents')
def list_uploaded_documents():
    return get_all_documents_in_store()



# Generate response to user query
@app.post('/chat')
def chat(query: QueryInput):
    # Retrieve the session_id from the query
    session_id = query.session_id or str(uuid.uuid4())

    logging.info(f"Session ID: {session_id}, User Query: {query.question}")
    
    # Use the session_id to retrieve chat history
    chat_history = get_chat_history(session_id)

    # Get the RAG chain with the appropriate model
    rag_chain = get_rag_chain(model=query.model.value)

    # Invoke the rag chain with the query and chat history
    response = rag_chain.invoke({"input": query.question, "chat_history":chat_history})
    response_text = response['answer']

    insert_application_logs(session_id=session_id,
                            user_query=query.question,
                            response=response_text,
                            model=query.model.value)
    
    logging.info(f"Session ID: {session_id}, AI response: {response_text}")

    return QueryResponse(answer=response_text, session_id=session_id, model=query.model) # Why though?