#!/usr/bin/env python3
"""
collect.py — collects review comments з GitHub PR and saves them to reviews.json

Usage:
    python collect.py --repo owner/repo-name
    python collect.py --repo owner/repo-name --state all
    python collect.py --repo owner/repo-name --state closed --output my_reviews.json

Setup:
    pip install PyGithub python-dotenv
    Create .env file with GITHUB_TOKEN=your_token_here
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from github import Github
from github.GithubException import GithubException

load_dotenv()


def collect_pr_comments(
    repo_name: str,
    state: str = "all",
    output_file: str = "reviews.json",
) -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found. Add it to .env file.")

    g = Github(token)

    print(f"Connecting to repository: {repo_name}")
    try:
        repo = g.get_repo(repo_name)
    except GithubException as e:
        raise ValueError(f"Cannot access repo '{repo_name}': {e.data.get('message', str(e))}")

    print(f"Fetching PRs (state={state})...")
    pulls = repo.get_pulls(state=state, sort="updated", direction="desc")

    # Load existing data to allow incremental updates
    existing_comments: dict[str, dict] = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
            existing_comments = {c["id"]: c for c in existing.get("comments", [])}
        print(f"Loaded {len(existing_comments)} existing comments from {output_file}")

    collected: list[dict] = []
    pr_count = 0
    comment_count = 0
    skipped_count = 0

    for pr in pulls:
        pr_count += 1
        print(f"  PR #{pr.number}: {pr.title[:60]}...", end="\r")

        try:
            review_comments = pr.get_review_comments()

            for comment in review_comments:
                comment_id = str(comment.id)

                # Skip if already collected (incremental mode)
                if comment_id in existing_comments:
                    skipped_count += 1
                    continue

                # Skip empty or bot comments
                body = comment.body.strip()
                if not body or len(body) < 10:
                    continue

                collected.append({
                    "id": comment_id,
                    "repo": repo_name,
                    "pr_number": pr.number,
                    "pr_title": pr.title,
                    "pr_state": pr.state,
                    "file": comment.path,
                    "line": comment.original_line or comment.line,
                    "diff_hunk": comment.diff_hunk,
                    "body": body,
                    "created_at": comment.created_at.isoformat(),
                    "collected_at": datetime.utcnow().isoformat(),
                })
                comment_count += 1

        except GithubException as e:
            print(f"\n  Warning: could not fetch comments for PR #{pr.number}: {e}")
            continue

    print(f"\nProcessed {pr_count} PRs")
    print(f"New comments collected: {comment_count}")
    print(f"Skipped (already in file): {skipped_count}")

    # Merge new comments with existing
    all_comments = list(existing_comments.values()) + collected

    output = {
        "meta": {
            "repo": repo_name,
            "total_comments": len(all_comments),
            "last_updated": datetime.utcnow().isoformat(),
        },
        "comments": all_comments,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_comments)} total comments to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GitHub PR review comments for AI reviewer training"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository in format owner/repo-name (e.g. myorg/mobile-app)",
    )
    parser.add_argument(
        "--state",
        choices=["open", "closed", "all"],
        default="all",
        help="PR state to collect from (default: all)",
    )
    parser.add_argument(
        "--output",
        default="reviews.json",
        help="Output JSON file path (default: reviews.json)",
    )

    args = parser.parse_args()
    collect_pr_comments(
        repo_name=args.repo,
        state=args.state,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()