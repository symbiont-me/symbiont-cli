# TODO to be completed
from qdrant_client import QdrantClient

client = QdrantClient("localhost:6333")

collections = client.get_collections()
print(collections)
collection_names = [
    "my_documents",
]

for name in collection_names:
    client.delete_collection(name)
