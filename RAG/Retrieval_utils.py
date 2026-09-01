import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def set_retrieval(db_path: str):
    """Initialize and return the ChromaDB client and QA collection."""
    collection_name = "qa_collection_with_answer"

    model_name = 'dragonkue/BGE-m3-ko'
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

    return client, collection

def get_most_similar_answer(collection, query: str, filter_source: str, filter_category: str):
    """Return the answer associated with the most similar filtered query."""
    if filter_source == "경찰청":
        return "관련 정보를 제공해 드리고 싶지만, 경찰청 업무와 관련된 내용이라 도움을 드리기 어렵습니다."
    
    results = collection.query(
        query_texts=[query],
        n_results=1,
        where={"category": filter_category}
    )

    if not results['ids'] or not results['ids'][0]:
        return "죄송합니다. 해당 조건에 맞는 결과를 찾을 수 없습니다."
        
    most_similar_q = results['documents'][0][0]
    corresponding_a = results['metadatas'][0][0]['answer']
    
    return corresponding_a
