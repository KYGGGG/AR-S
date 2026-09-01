import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from pathlib import Path
import os

model_name = 'dragonkue/BGE-m3-ko'
PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
MODULE_DIR = PROJECT_DIR / "RAG"
db_path = MODULE_DIR / "db_path"

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
client = chromadb.PersistentClient(path=str(db_path))

# Load the existing classified QA dataset.
file_path_existing = MODULE_DIR / 'classified_qa_data.csv'
df_existing = pd.read_csv(file_path_existing, encoding='utf-8-sig')

# Load new QA records to append.
file_path_new = MODULE_DIR / 'add_qa_data.csv'
df_new = pd.read_csv(file_path_new, encoding='cp949')

# Merge and normalize both datasets.
df = pd.concat([df_existing, df_new], ignore_index=True)
df = df.dropna(subset=['Q', 'A'])

documents = df['Q'].tolist()
# Generate stable IDs from the merged dataframe index.
ids = [str(i) for i in df.index]

metadatas = [
    {
        'source': row['Predicted_Source'],
        'category': row['Predicted_Category'],
        'answer': row['A'] 
    }
    for index, row in df.iterrows()
]

# Reuse the collection when it already exists.
collection_name = "qa_collection_with_answer"
collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

# Append only records that are not already stored.
existing_count = collection.count()

documents_to_add = documents[existing_count:]
metadatas_to_add = metadatas[existing_count:]
ids_to_add = ids[existing_count:]

chunk_size = 1000
print("ChromaDB 컬렉션에 새로운 데이터를 청크 단위로 추가 중...")

# Add new records in manageable chunks.
for i in tqdm(range(0, len(documents_to_add), chunk_size)):
    chunk_docs = documents_to_add[i:i + chunk_size]
    chunk_metadatas = metadatas_to_add[i:i + chunk_size]
    chunk_ids = ids_to_add[i:i + chunk_size]

    collection.add(
        documents=chunk_docs,
        metadatas=chunk_metadatas,
        ids=chunk_ids
    )

print("\n데이터 추가 완료.")
print(f"총 {collection.count()}개의 문서가 컬렉션에 저장되었습니다.")
