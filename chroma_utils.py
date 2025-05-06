from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing_extensions import List
from langchain_core.documents import Document

from langchain_core.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader



# Initialize Vector Store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               length_function=len)
embedding_model = OpenAIEmbeddings()
vector_store = Chroma(persist_directory='./chroma_db',
                      embedding_function=embedding_model)



def load_and_split_document(filepath:str) -> List[Document]:
    # Load Document
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".html"):
        loader = UnstructuredHTMLLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")
    
    documents = loader.load()
    # Split document
    return text_splitter.split_documents(documents)


def index_document_to_chroma(filepath: str, file_id: int) -> bool:
    # File ID is to link these splits back to the files in the document store

    try:
        # Load and Split document
        documents = load_and_split_document(filepath)

        # Add metadata to each split
        for document in documents:
            document.metadata['file_id'] = file_id

        # Index all splits
        vector_store.add_documents(documents)
        return True

    except Exception as e:
        print(f"Error indexing document: {e}")
        return False



def delete_file_from_chroma(file_id: int) -> bool:
    try:
        # Pick all chunks with metadata as provided file_id and delete them
        delete_documents = vector_store.get(where={'file_id': file_id})
        print(f"Found {len(delete_documents)} entries in the vector db for file id: {file_id}")

        vector_store.delete(ids=delete_documents['ids'])
        print(f"Deleted all documents with file_id {file_id}")
        return True
    
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False