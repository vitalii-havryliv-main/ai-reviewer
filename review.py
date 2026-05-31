#!/usr/bin/env python3
"""
review.py — reviews your current branch diff against base using team's past PR comments
            + full file context per hunk to reduce false positives

Usage:
    python review.py
    python review.py --base 0.2.x
    python review.py --base main --db db/chroma --top 10

Setup:
    Make sure OPENAI_API_KEY and ANTHROPIC_API_KEY are in your .env file
    Run collect.py and embed.py first to build the knowledge base
"""

import os
import re
import json
import argparse
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import anthropic

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "pr_reviews"
CLAUDE_MODEL = "claude-sonnet-4-6"

# How many lines around each hunk to include as context when file is large
HUNK_CONTEXT_LINES = 100
# Max file size to include fully (in lines)
MAX_FULL_FILE_LINES = 300
# Minimum similarity score to include a past comment
MIN_SIMILARITY = 0.5
# How many past comments to retrieve per chunk
TOP_N_COMMENTS = 10


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True,
    )
    branch = result.stdout.strip()
    if not branch:
        raise ValueError("Could not detect current git branch. Are you in a git repo?")
    return branch


def get_diff(base_branch: str, current_branch: str) -> str:
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...{current_branch}"],
        capture_output=True, text=True,
    )
    diff = result.stdout.strip()
    if not diff:
        raise ValueError(
            f"No diff found between '{base_branch}' and '{current_branch}'. "
            "Make sure you have committed your changes."
        )
    return diff


def get_repo_root() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Diff parsing — split into per-file chunks with hunk line numbers
# ---------------------------------------------------------------------------

def parse_diff_by_file(diff: str) -> list[dict]:
    """
    Split diff into per-file entries.
    Returns list of:
      { "file": str, "diff": str, "hunks": [{"start": int, "end": int}] }
    """
    files = []
    current: dict | None = None

    hunk_header_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            if current:
                files.append(current)
            # Extract b/ path
            match = re.search(r" b/(.+)$", line)
            file_path = match.group(1) if match else "unknown"
            current = {"file": file_path, "diff": line + "\n", "hunks": []}
        elif current is not None:
            current["diff"] += line + "\n"
            m = hunk_header_re.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2)) if m.group(2) else 1
                current["hunks"].append({"start": start, "end": start + length})

    if current:
        files.append(current)

    return files


# ---------------------------------------------------------------------------
# Full file context
# ---------------------------------------------------------------------------

def read_file_context(file_path: str, hunks: list[dict], repo_root: str) -> str | None:
    """
    Read the full file or ±HUNK_CONTEXT_LINES around each hunk.
    Returns None if file doesn't exist (deleted file).
    """
    full_path = os.path.join(repo_root, file_path)
    if not os.path.exists(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    total_lines = len(lines)

    # Small file — return everything
    if total_lines <= MAX_FULL_FILE_LINES:
        content = "".join(lines)
        return f"=== FULL FILE: {file_path} ({total_lines} lines) ===\n{content}"

    # Large file — return ±HUNK_CONTEXT_LINES around each hunk
    segments = []
    covered: set[int] = set()

    for hunk in hunks:
        lo = max(0, hunk["start"] - HUNK_CONTEXT_LINES - 1)
        hi = min(total_lines, hunk["end"] + HUNK_CONTEXT_LINES)
        seg_lines = []
        for i in range(lo, hi):
            if i not in covered:
                seg_lines.append(f"{i+1:4d} | {lines[i]}")
                covered.add(i)
        if seg_lines:
            segments.append(
                f"--- {file_path} lines {lo+1}–{hi} ---\n" + "".join(seg_lines)
            )

    if not segments:
        return None

    return (
        f"=== CONTEXT: {file_path} ({total_lines} lines total, "
        f"showing {len(covered)} lines around changes) ===\n"
        + "\n".join(segments)
    )


# ---------------------------------------------------------------------------
# RAG — find similar past comments
# ---------------------------------------------------------------------------

def search_similar_comments(
    diff_chunk: str,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    top_n: int = TOP_N_COMMENTS,
) -> list[dict]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=diff_chunk[:6000],
    )
    diff_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[diff_vector],
        n_results=min(top_n, collection.count()),
        include=["metadatas", "distances"],
    )

    comments = []
    seen_bodies: set[str] = set()

    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - distance
        if similarity < MIN_SIMILARITY:
            continue
        body = metadata.get("body", "").strip()
        # Dedup identical comments
        key = body[:100]
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        comments.append({
            "file": metadata.get("file", ""),
            "body": body,
            "similarity": round(similarity, 3),
        })

    return comments


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(
    file_path: str,
    diff_chunk: str,
    file_context: str | None,
    similar_comments: list[dict],
) -> str:
    comments_text = ""
    for comment in similar_comments:
        comments_text += f"- [{comment['file']}] {comment['body']}\n"

    context_section = ""
    if file_context:
        context_section = f"""
--- FULL FILE CONTEXT (use this to verify your findings) ---
{file_context}
--- END FILE CONTEXT ---
"""

    return f"""You are a senior code reviewer doing a pre-PR review.

You have access to:
1. The git diff for this file
2. The full file context (or relevant lines) so you can verify findings
3. Past review comments from this team on similar code

IMPORTANT RULES:
- CRITICAL severity requires evidence from the FULL FILE CONTEXT, not just the diff hunk
- If you cannot see the full function in the diff — check the file context before marking CRITICAL
- Do NOT invent references to past comments — only use the ones provided below
- Do NOT flag intentional design decisions as bugs without evidence
- If the guard/check exists elsewhere in the file, do NOT flag it as missing

--- PAST TEAM REVIEW COMMENTS (use as style guidance only) ---
{comments_text if comments_text else "No similar past comments found."}
--- END PAST COMMENTS ---

--- DIFF ---
{diff_chunk}
--- END DIFF ---
{context_section}

Respond with a JSON array of findings. Each finding:
{{
  "severity": "critical" | "important" | "minor",
  "confidence": "high" | "medium" | "low",
  "file": "filename.ts",
  "line": "approximate line number or range",
  "title": "short title",
  "problem": "what is wrong and why it matters",
  "fix": "how to fix it (code snippet if helpful)",
  "evidence": "exact quote from full file context that confirms this issue exists"
}}

Rules:
- "critical": real bug, data loss, crash, security issue — only with confidence "high" and evidence from full file
- "important": wrong pattern, missing type safety, performance issue
- "minor": style, naming, missing comment, small improvement
- If no issues found: return empty array []
- Return ONLY the JSON array, no other text"""


# ---------------------------------------------------------------------------
# Dedup findings across chunks
# ---------------------------------------------------------------------------

def dedup_findings(findings: list[dict]) -> list[dict]:
    """
    Remove duplicate findings across chunks.
    Two findings are duplicates if same file + very similar title.
    """
    seen: set[str] = set()
    result = []
    for f in findings:
        key = f"{f.get('file', '')}::{f.get('title', '')[:50].lower()}"
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {
    "critical": "🔴",
    "important": "🟠",
    "minor": "🟢",
}

SEVERITY_ORDER = {"critical": 0, "important": 1, "minor": 2}
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def findings_to_markdown(
    findings: list[dict],
    current_branch: str,
    base_branch: str,
    diff_lines: int,
) -> str:
    if not findings:
        return (
            f"# Code Review\n"
            f"**Branch:** `{current_branch}` → `{base_branch}`\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"**Diff size:** {diff_lines} lines\n\n"
            f"---\n\n"
            f"✅ No issues found.\n"
        )

    # Sort: severity → confidence
    sorted_findings = sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.get("severity", "minor"), 2),
            CONFIDENCE_ORDER.get(f.get("confidence", "low"), 2),
        ),
    )

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for f in sorted_findings:
        file_key = f.get("file", "unknown")
        by_file.setdefault(file_key, []).append(f)

    # Summary counts
    critical = sum(1 for f in sorted_findings if f.get("severity") == "critical")
    important = sum(1 for f in sorted_findings if f.get("severity") == "important")
    minor = sum(1 for f in sorted_findings if f.get("severity") == "minor")

    lines = [
        f"# Code Review",
        f"**Branch:** `{current_branch}` → `{base_branch}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Diff size:** {diff_lines} lines",
        f"**Findings:** 🔴 {critical} critical · 🟠 {important} important · 🟢 {minor} minor",
        "",
        "---",
        "",
    ]

    for file_path, file_findings in by_file.items():
        lines.append(f"## `{file_path}`")
        lines.append("")

        for f in file_findings:
            severity = f.get("severity", "minor")
            confidence = f.get("confidence", "low")
            emoji = SEVERITY_EMOJI.get(severity, "🟢")
            label = severity.upper()
            conf_note = f" _(confidence: {confidence})_" if confidence != "high" else ""

            lines.append(f"### {emoji} {label} — {f.get('title', 'Issue')}{conf_note}")
            lines.append("")

            if f.get("line"):
                lines.append(f"**Line:** {f['line']}")
                lines.append("")

            lines.append(f"**Problem:** {f.get('problem', '')}")
            lines.append("")

            if f.get("fix"):
                lines.append(f"**Fix:**")
                lines.append(f"```")
                lines.append(f.get("fix", ""))
                lines.append(f"```")
                lines.append("")

            if f.get("evidence") and confidence != "high":
                lines.append(f"**Evidence:** _{f.get('evidence', '')}_")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    if critical > 0:
        lines.append(f"🔴 **{critical} critical issue(s)** — must fix before merge.")
    if important > 0:
        lines.append(f"🟠 **{important} important issue(s)** — should fix before merge.")
    if minor > 0:
        lines.append(f"🟢 **{minor} minor issue(s)** — consider fixing for code quality.")
    lines.append("")
    lines.append(
        "> ⚠️ Always verify CRITICAL findings manually. "
        "Reviewer has full file context but may miss product intent from PR description."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main review flow
# ---------------------------------------------------------------------------

def review_diff(
    base_branch: str = "main",
    db_path: str = "db/chroma",
    top_n: int = TOP_N_COMMENTS,
) -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at '{db_path}'. Run embed.py first.")

    current_branch = get_current_branch()
    repo_root = get_repo_root()

    print(f"Branch:    {current_branch}")
    print(f"Base:      {base_branch}")
    print(f"Repo root: {repo_root}")
    print(f"Diffing {base_branch}...{current_branch}\n")

    diff = get_diff(base_branch, current_branch)
    diff_lines = diff.count("\n")
    print(f"Diff size: {diff_lines} lines")

    file_diffs = parse_diff_by_file(diff)
    print(f"Changed files: {len(file_diffs)}")

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Knowledge base: {collection.count()} comments\n")

    openai_client = OpenAI(api_key=openai_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    all_findings: list[dict] = []

    for i, file_entry in enumerate(file_diffs, 1):
        file_path = file_entry["file"]
        file_diff = file_entry["diff"]
        hunks = file_entry["hunks"]

        print(f"[{i}/{len(file_diffs)}] {file_path} ({len(hunks)} hunk(s))")

        # Skip binary / lock files
        skip_extensions = {".lock", ".pbxproj", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico"}
        if any(file_path.endswith(ext) for ext in skip_extensions):
            # Still include pbxproj in review but skip context reading
            if file_path.endswith(".lock"):
                print("  Skipping lock file")
                continue

        # Read full file context
        file_context = read_file_context(file_path, hunks, repo_root)
        if file_context:
            context_lines = file_context.count("\n")
            print(f"  File context: {context_lines} lines")
        else:
            print(f"  File context: not available (deleted or binary)")

        # Find similar past comments
        similar_comments = search_similar_comments(file_diff, openai_client, collection, top_n)
        print(f"  Similar past comments: {len(similar_comments)}")

        # Build prompt and call Claude
        prompt = build_prompt(file_path, file_diff, file_context, similar_comments)

        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if Claude wrapped the JSON
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            findings = json.loads(raw)
            if isinstance(findings, list):
                # Downgrade low-confidence criticals to important
                for f in findings:
                    if f.get("severity") == "critical" and f.get("confidence") != "high":
                        f["severity"] = "important"
                        f["title"] = f.get("title", "") + " (downgraded: low confidence)"
                all_findings.extend(findings)
                print(f"  Findings: {len(findings)}")
            else:
                print(f"  Unexpected response format, skipping")

        except json.JSONDecodeError as e:
            print(f"  Could not parse JSON response: {e}")
            print(f"  Raw response: {raw[:200]}")
        except Exception as e:
            print(f"  Error reviewing {file_path}: {e}")

    # Dedup across all files
    deduped = dedup_findings(all_findings)
    removed = len(all_findings) - len(deduped)
    if removed:
        print(f"\nDeduped {removed} duplicate finding(s)")

    # Render markdown
    markdown = findings_to_markdown(deduped, current_branch, base_branch, diff_lines)

    # Save to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_filename = f"review_{timestamp}.md"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n✅ Review saved to: {output_filename}")
    print(f"   {len(deduped)} findings total\n")
    print(markdown)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI code reviewer based on your team's PR history"
    )
    parser.add_argument("--base", default="main", help="Base branch to diff against (default: main)")
    parser.add_argument("--db", default="db/chroma", help="ChromaDB path (default: db/chroma)")
    parser.add_argument("--top", type=int, default=TOP_N_COMMENTS, help="Past comments to retrieve per file")

    args = parser.parse_args()
    review_diff(base_branch=args.base, db_path=args.db, top_n=args.top)


if __name__ == "__main__":
    main()