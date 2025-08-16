import argparse
from qdrant_client import QdrantClient


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Remove a collection from Qdrant")
    parser.add_argument(
        "collection_name", type=str, help="Name of the collection to remove"
    )
    args = parser.parse_args()

    client = QdrantClient("localhost:6333")

    collection_name = args.collection_name

    try:
        print(f"Attempting to delete collection: {collection_name}")
        client.delete_collection(collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        print(f"Failed to delete collection: {collection_name}. Error: {e}")


if __name__ == "__main__":
    main()
