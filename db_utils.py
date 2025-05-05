import sqlite3
from langchain_chroma import Chroma

from pydantic_models import DocumentInfo
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import List


DB_NAME = 'rag_app.db'

def connect_to_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn



# Application Logs
def create_application_logs():
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS application_logs 
                   (id PRIMARY KEY AUTOINCREMENT,
                   session_id TEXT,
                   user_query TEXT,
                   response TEXT,
                   model TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.close()
    return


def insert_application_log(session_id, user_query, response, model):
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("""INSERT INTO application_logs
                   (session_id, user_query, response, model) VALUES (?, ?, ?, ?)
                   """, (session_id, user_query, response, model,))
    conn.commit()
    conn.close()
    return 


def get_chat_history(session_id: str) -> List[AIMessage|HumanMessage]:
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute(f"""SELECT user_query, response
                        FROM application_logs
                        WHERE session_id={session_id}
                    """)
    messages=[]
    for record in cursor.fetchall():
        human_message = record['user_query']
        ai_message = record['response']

        human_message = HumanMessage(content=human_message)
        ai_message = AIMessage(content=ai_message)
        messages.extend([human_message, ai_message])
    
    conn.close()
    return messages



# Document Store
def create_document_store():
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS document_store 
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   filename TEXT,
                   upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.close()
    return


def insert_file_to_document_store(filename) -> int:
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute(f"""INSERT INTO document_store
                   (filename) VALUES (?)""", (filename,))

    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_file_from_document_store(file_id) -> bool:
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute(f"""DELETE FROM document_store WHERE id= ?""", (file_id,))
    conn.commit()
    conn.close()
    return True


def get_all_documents_in_store() -> List:
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("""SELECT id, filename, upload_timestamp from document_store""")
    
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]