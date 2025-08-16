from qdrant_client import QdrantClient


def main():
    client = QdrantClient("localhost:6333")

    try:
        collections = client.get_collections()
        # Print the collection names
        for collection in collections.collections:
            print(collection.name)
    except Exception as e:
        print(f"Failed Error: {e}")


if __name__ == "__main__":
    main()
