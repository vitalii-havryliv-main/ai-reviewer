#!/usr/bin/env python3
"""
embed.py — converts PR review comments into vectors and stores in ChromaDB

Usage:
    python embed.py
    python embed.py --input data/reviews.json
    python embed.py --input data/reviews.json --db db/chroma

Setup:
    pip install -r requirements.txt
    Make sure OPENAI_API_KEY is in your .env file
"""

import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "pr_reviews"
BATCH_SIZE = 100  # OpenAI allows up to 2048, but 100 is safe and trackable


def build_text(comment: dict) -> str:
    """
    Combine file + diff_hunk + body into one string for embedding.
    The vector needs to capture both the code context and the feedback.
    """
    parts = []

    if comment.get("file"):
        parts.append(f"File: {comment['file']}")

    if comment.get("diff_hunk"):
        parts.append(f"Code:\n{comment['diff_hunk']}")

    if comment.get("body"):
        parts.append(f"Review comment: {comment['body']}")

    return "\n\n".join(parts)


def embed_comments(
    input_file: str = "data/reviews.json",
    db_path: str = "db/chroma",
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to .env file.")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"{input_file} not found. Run collect.py first."
        )

    # Load comments
    print(f"Loading comments from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    comments = data.get("comments", [])
    if not comments:
        print("No comments found in input file.")
        return

    print(f"Loaded {len(comments)} comments")

    # Connect to ChromaDB
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )

    # Check which comments are already embedded (incremental)
    existing_ids = set(collection.get()["ids"])
    print(f"Already embedded: {len(existing_ids)} comments")

    new_comments = [c for c in comments if str(c["id"]) not in existing_ids]
    if not new_comments:
        print("Nothing new to embed. Database is up to date.")
        return

    print(f"New comments to embed: {len(new_comments)}")

    # Initialize OpenAI
    openai_client = OpenAI(api_key=api_key)

    # Process in batches
    total_batches = (len(new_comments) + BATCH_SIZE - 1) // BATCH_SIZE
    total_embedded = 0

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = new_comments[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} comments)...")

        # Build texts for embedding
        texts = [build_text(c) for c in batch]
        ids = [str(c["id"]) for c in batch]
        metadatas = [
            {
                "repo": c.get("repo", ""),
                "pr_number": str(c.get("pr_number", "")),
                "pr_state": c.get("pr_state", ""),
                "file": c.get("file", ""),
                "line": str(c.get("line", "")),
                "body": c.get("body", "")[:500],  # ChromaDB metadata limit
            }
            for c in batch
        ]

        # Get embeddings from OpenAI
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]

        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_embedded += len(batch)
        print(f"  Embedded {total_embedded}/{len(new_comments)}")

    print(f"\nDone. Total comments in database: {collection.count()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed PR review comments into ChromaDB vector store"
    )
    parser.add_argument(
        "--input",
        default="data/reviews.json",
        help="Input JSON file from collect.py (default: data/reviews.json)",
    )
    parser.add_argument(
        "--db",
        default="db/chroma",
        help="ChromaDB storage path (default: db/chroma)",
    )

    args = parser.parse_args()
    embed_comments(
        input_file=args.input,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
