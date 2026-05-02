#!/usr/bin/env python3
"""
review.py — reviews your current branch diff against main using team's past PR comments

Usage:
    python review.py
    python review.py --base develop
    python review.py --base main --db db/chroma --top 20

Setup:
    Make sure OPENAI_API_KEY and ANTHROPIC_API_KEY are in your .env file
    Run collect.py and embed.py first to build the knowledge base
"""

import os
import json
import argparse
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import anthropic

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "pr_reviews"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def get_current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    if not branch:
        raise ValueError("Could not detect current git branch. Are you in a git repo?")
    return branch


def get_diff(base_branch: str, current_branch: str) -> str:
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...{current_branch}"],
        capture_output=True,
        text=True,
    )
    diff = result.stdout.strip()
    if not diff:
        raise ValueError(
            f"No diff found between '{base_branch}' and '{current_branch}'. "
            "Make sure you have committed your changes."
        )
    return diff


def chunk_diff(diff: str, max_chars: int = 8000) -> list[str]:
    """
    Split large diffs into chunks so they fit in context window.
    Splits on file boundaries (lines starting with 'diff --git').
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for line in diff.split("\n"):
        if line.startswith("diff --git") and current_size > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = len(line)
        else:
            current_chunk.append(line)
            current_size += len(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def search_similar_comments(
    diff: str,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    top_n: int = 20,
) -> list[dict]:
    """
    Convert diff to vector and find most similar past review comments.
    """
    # Truncate diff for embedding — OpenAI has token limits
    diff_for_embedding = diff[:6000]

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=diff_for_embedding,
    )
    diff_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[diff_vector],
        n_results=min(top_n, collection.count()),
        include=["metadatas", "distances"],
    )

    comments = []
    for metadata, distance in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance  # cosine distance → similarity score
        if similarity > 0.3:  # filter out low-relevance comments
            comments.append({
                "file": metadata.get("file", ""),
                "body": metadata.get("body", ""),
                "similarity": round(similarity, 3),
            })

    return comments


def build_prompt(diff: str, similar_comments: list[dict]) -> str:
    comments_text = ""
    for i, comment in enumerate(similar_comments, 1):
        comments_text += f"{i}. [{comment['file']}] {comment['body']}\n"

    return f"""You are a senior code reviewer. Your team has left these review comments on similar code in the past:

--- PAST REVIEW COMMENTS FROM YOUR TEAM ---
{comments_text}
--- END OF PAST COMMENTS ---

Now review this diff and provide feedback based on your team's patterns and general best practices.

--- DIFF TO REVIEW ---
{diff}
--- END OF DIFF ---

Provide a structured code review. For each issue found:
- Mention the file and approximate line
- Explain what the problem is
- Suggest how to fix it

Focus on real issues. If the code looks good — say so. Do not invent problems.
Group feedback by file. Be direct and concise."""


def review_diff(
    base_branch: str = "main",
    db_path: str = "db/chroma",
    top_n: int = 20,
) -> None:
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env")

    # Check ChromaDB exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. Run embed.py first."
        )

    # Get current branch and diff
    current_branch = get_current_branch()
    print(f"Branch:  {current_branch}")
    print(f"Base:    {base_branch}")
    print(f"Diffing {base_branch}...{current_branch}\n")

    diff = get_diff(base_branch, current_branch)
    diff_lines = diff.count("\n")
    print(f"Diff size: {diff_lines} lines")

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Knowledge base: {collection.count()} comments\n")

    # Initialize clients
    openai_client = OpenAI(api_key=openai_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    # Split large diffs into chunks
    chunks = chunk_diff(diff)
    print(f"Processing {len(chunks)} chunk(s)...\n")

    all_reviews = []

    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"--- Chunk {i}/{len(chunks)} ---")

        # Find similar past comments
        print("Searching knowledge base for similar comments...")
        similar_comments = search_similar_comments(
            chunk, openai_client, collection, top_n
        )
        print(f"Found {len(similar_comments)} relevant past comments\n")

        # Build prompt and call Claude
        prompt = build_prompt(chunk, similar_comments)

        print("Asking Claude to review...\n")
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        review_text = response.content[0].text
        all_reviews.append(review_text)

    # Print final review
    separator = "\n" + "=" * 60 + "\n"
    print(separator)
    print("CODE REVIEW")
    print(separator)
    print(separator.join(all_reviews))
    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI code reviewer based on your team's PR history"
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base branch to diff against (default: main)",
    )
    parser.add_argument(
        "--db",
        default="db/chroma",
        help="ChromaDB path (default: db/chroma)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of similar comments to use as context (default: 20)",
    )

    args = parser.parse_args()
    review_diff(
        base_branch=args.base,
        db_path=args.db,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
