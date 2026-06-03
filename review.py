#!/usr/bin/env python3
"""
review.py — AI code reviewer with agentic file lookup for cross-file verification

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

EMBEDDING_MODEL  = "text-embedding-3-small"
COLLECTION_NAME  = "pr_reviews"
CLAUDE_MODEL     = "claude-sonnet-4-6"

HUNK_CONTEXT_LINES  = 100   # lines around each hunk for large files
MAX_FULL_FILE_LINES = 300   # files smaller than this are included fully
MIN_SIMILARITY      = 0.5   # RAG threshold
TOP_N_COMMENTS      = 10    # past comments per file
MAX_LOOKUP_ROUNDS   = 3     # max agentic file-lookup rounds per file
MAX_LOOKUP_FILES    = 5     # max extra files Claude can request per file


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_current_branch() -> str:
    r = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    branch = r.stdout.strip()
    if not branch:
        raise ValueError("Could not detect current git branch. Are you in a git repo?")
    return branch


def get_diff(base_branch: str, current_branch: str) -> str:
    r = subprocess.run(
        ["git", "diff", f"{base_branch}...{current_branch}"],
        capture_output=True, text=True,
    )
    diff = r.stdout.strip()
    if not diff:
        raise ValueError(
            f"No diff found between '{base_branch}' and '{current_branch}'. "
            "Make sure you have committed your changes."
        )
    return diff


def get_repo_root() -> str:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------

def load_project_config(config_path: str | None) -> dict:
    """Load project.json config if provided. Returns empty dict if not."""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Project config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_always_include_files(
    file_path: str,
    project_config: dict,
    repo_root: str,
) -> dict[str, str]:
    """
    Auto-load files defined in always_include when diff file matches a pattern.
    """
    always_include = project_config.get("always_include", [])
    extra: dict[str, str] = {}

    for rule in always_include:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        if re.search(pattern, file_path, re.IGNORECASE):
            for inc_path in rule.get("files", []):
                if inc_path not in extra:
                    content = read_file(inc_path, repo_root, hunks=None)
                    if content:
                        extra[inc_path] = content

    return extra


def resolve_import_path(raw_path: str, project_config: dict) -> str:
    """Resolve import alias to real path using project config."""
    aliases = project_config.get("import_aliases", {})
    for alias, real in aliases.items():
        if raw_path.startswith(alias):
            return raw_path.replace(alias, real, 1)
    return raw_path


def build_project_context_section(project_config: dict) -> str:
    """Build project context string to inject into prompt."""
    if not project_config:
        return ""

    parts = []

    name = project_config.get("name", "")
    if name:
        parts.append(f"Project: {name}")

    stack = project_config.get("stack", {})
    if stack:
        langs = ", ".join(stack.get("languages", []))
        frameworks = ", ".join(stack.get("frameworks", []))
        libs = stack.get("key_libs", [])
        if langs:
            parts.append(f"Languages: {langs}")
        if frameworks:
            parts.append(f"Frameworks: {frameworks}")
        if libs:
            parts.append("Key libraries:")
            for lib in libs:
                parts.append(f"  - {lib}")

    structure = project_config.get("structure", "")
    if structure:
        parts.append("\nProject structure:\n" + structure.strip())

    intentional = project_config.get("intentional_patterns", "")
    if intentional:
        parts.append("\nINTENTIONAL PATTERNS (do NOT flag these as bugs):\n" + intentional.strip())

    rules = project_config.get("review_rules", "")
    if rules:
        parts.append("\nPROJECT REVIEW RULES (apply these as priorities):\n" + rules.strip())

    additional = project_config.get("additional_context", "")
    if additional:
        parts.append("\nADDITIONAL PROJECT CONTEXT:\n" + additional.strip())

    if not parts:
        return ""

    return "--- PROJECT CONTEXT ---\n" + "\n".join(parts) + "\n--- END PROJECT CONTEXT ---"



# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

def parse_diff_by_file(diff: str) -> list[dict]:
    files = []
    current: dict | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            if current:
                files.append(current)
            match = re.search(r" b/(.+)$", line)
            file_path = match.group(1) if match else "unknown"
            current = {"file": file_path, "diff": line + "\n", "hunks": []}
        elif current is not None:
            current["diff"] += line + "\n"
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2)) if m.group(2) else 1
                current["hunks"].append({"start": start, "end": start + length})

    if current:
        files.append(current)
    return files


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_file(file_path: str, repo_root: str, hunks: list[dict] | None = None) -> str | None:
    """
    Read a file from repo. If hunks provided and file is large — return context only.
    If hunks is None (lookup request) — return full file up to 500 lines.
    """
    full_path = os.path.join(repo_root, file_path)
    if not os.path.exists(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    total = len(lines)

    # For lookup requests — return full file capped at 500 lines
    if hunks is None:
        cap = 500
        content = "".join(lines[:cap])
        truncated = f" (truncated to {cap} lines)" if total > cap else ""
        return f"=== FILE: {file_path} ({total} lines{truncated}) ===\n{content}"

    # For diff file — full if small, context if large
    if total <= MAX_FULL_FILE_LINES:
        return f"=== FULL FILE: {file_path} ({total} lines) ===\n{''.join(lines)}"

    segments = []
    covered: set[int] = set()
    for hunk in hunks:
        lo = max(0, hunk["start"] - HUNK_CONTEXT_LINES - 1)
        hi = min(total, hunk["end"] + HUNK_CONTEXT_LINES)
        seg = []
        for i in range(lo, hi):
            if i not in covered:
                seg.append(f"{i+1:4d} | {lines[i]}")
                covered.add(i)
        if seg:
            segments.append(f"--- {file_path} lines {lo+1}–{hi} ---\n" + "".join(seg))

    if not segments:
        return None

    return (
        f"=== CONTEXT: {file_path} ({total} lines total, "
        f"showing {len(covered)} lines around changes) ===\n"
        + "\n".join(segments)
    )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def search_similar_comments(
    diff_chunk: str,
    openai_client: OpenAI,
    collection: chromadb.Collection,
    top_n: int = TOP_N_COMMENTS,
) -> list[dict]:
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=diff_chunk[:6000])
    vec = r.data[0].embedding

    results = collection.query(
        query_embeddings=[vec],
        n_results=min(top_n, collection.count()),
        include=["metadatas", "distances"],
    )

    comments = []
    seen: set[str] = set()
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if 1 - dist < MIN_SIMILARITY:
            continue
        body = meta.get("body", "").strip()
        key = body[:100]
        if key in seen:
            continue
        seen.add(key)
        comments.append({"file": meta.get("file", ""), "body": body})

    return comments


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_initial_prompt(
    file_path: str,
    diff: str,
    file_context: str | None,
    similar_comments: list[dict],
    extra_files: dict[str, str],
    project_context: str = "",
) -> str:
    comments_text = "\n".join(
        f"- [{c['file']}] {c['body']}" for c in similar_comments
    ) or "No similar past comments found."

    context_section = ""
    if file_context:
        context_section = f"\n--- PRIMARY FILE CONTEXT ---\n{file_context}\n--- END ---\n"

    extra_section = ""
    if extra_files:
        parts = [f"\n--- ADDITIONAL FILE: {path} ---\n{content}\n--- END ---"
                 for path, content in extra_files.items()]
        extra_section = "\n".join(parts)

    project_section = ("\n" + project_context + "\n") if project_context else ""

    return f"""You are a senior code reviewer doing a pre-PR review.
{project_section}
STRICT SEVERITY RULES:
- "critical": real bug / data loss / crash / security. ONLY with confidence "high" AND evidence from the full file context. NEVER if the issue depends on logic in another file you haven't seen.
- "important": wrong pattern, missing type safety, performance issue, missing error handling
- "minor": style, naming, missing comment, small improvement. This includes: "add a comment", "rename variable", "test structure", "extract constant", "add TODO"
- If you cannot see the full function — do NOT mark CRITICAL. Request the file instead.
- If a guard/null-check exists elsewhere in the same file — do NOT flag it as missing.
- Do NOT invent references to past comments. Only use what is provided below.

PAST TEAM COMMENTS (style guidance only):
{comments_text}

DIFF TO REVIEW:
{diff}
{context_section}{extra_section}

If you need to see another file to verify a potential CRITICAL finding, respond with:
{{"action": "need_files", "files": ["path/to/file.ts", "path/to/other.kt"], "reason": "why you need them"}}

Otherwise respond with a JSON array of findings:
[
  {{
    "severity": "critical" | "important" | "minor",
    "confidence": "high" | "medium" | "low",
    "file": "{file_path}",
    "line": "line number or range",
    "title": "short title",
    "problem": "what is wrong and why it matters",
    "fix": "how to fix it, code snippet if helpful",
    "evidence": "exact quote from file context confirming the issue"
  }}
]

Return ONLY valid JSON — either the need_files object or the findings array. No other text."""


def build_followup_prompt(reason: str, extra_files: dict[str, str]) -> str:
    parts = [f"\n--- FILE: {path} ---\n{content}\n--- END ---"
             for path, content in extra_files.items()]
    files_text = "\n".join(parts)

    return f"""You requested these files to verify your findings: {reason}

{files_text}

Now provide your final review as a JSON array of findings.
Apply the same STRICT SEVERITY RULES as before.
Return ONLY the JSON array, no other text."""


# ---------------------------------------------------------------------------
# Agentic review loop
# ---------------------------------------------------------------------------

def agentic_review_file(
    file_path: str,
    file_diff: str,
    file_context: str | None,
    similar_comments: list[dict],
    repo_root: str,
    claude_client: anthropic.Anthropic,
    project_context: str = "",
    project_config: dict | None = None,
) -> list[dict]:
    """
    Review a single file with agentic file lookup.
    Claude can request up to MAX_LOOKUP_FILES extra files before giving final answer.
    """
    extra_files: dict[str, str] = {}
    fetched_files: set[str] = set()
    messages = []

    # Round 1 — initial review
    initial_prompt = build_initial_prompt(
        file_path, file_diff, file_context, similar_comments, extra_files,
        project_context=project_context,
    )
    messages.append({"role": "user", "content": initial_prompt})

    for round_num in range(1, MAX_LOOKUP_ROUNDS + 1):
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=3000,
            messages=messages,
        )
        raw = response.content[0].text.strip()

        # Strip code fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ⚠️  Could not parse JSON (round {round_num}): {raw[:150]}")
            return []

        # Claude wants more files
        if isinstance(parsed, dict) and parsed.get("action") == "need_files":
            requested = parsed.get("files", [])
            reason = parsed.get("reason", "")

            if round_num >= MAX_LOOKUP_ROUNDS:
                print(f"  ⚠️  Max lookup rounds reached, proceeding without extra files")
                # Force final answer
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "Max file lookups reached. Give your best final review now as a JSON array. If you cannot confirm a CRITICAL finding without the missing files, downgrade it to important."
                })
                continue

            # Read requested files
            newly_fetched: dict[str, str] = {}
            for req_path in requested[:MAX_LOOKUP_FILES]:
                if req_path in fetched_files:
                    continue
                # Normalize path — strip leading /
                clean_path = req_path.lstrip("/")
                # Resolve import aliases if project config provided
                if project_config:
                    clean_path = resolve_import_path(clean_path, project_config)
                content = read_file(clean_path, repo_root, hunks=None)
                if content:
                    newly_fetched[clean_path] = content
                    fetched_files.add(req_path)
                    print(f"  🔍 Fetched: {clean_path}")
                else:
                    print(f"  ❌ Not found: {clean_path}")

            if not newly_fetched:
                # Nothing found — force final answer
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "The requested files were not found in the repo. Give your final review as a JSON array without them. Downgrade any unverified CRITICAL to important."
                })
                continue

            extra_files.update(newly_fetched)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": build_followup_prompt(reason, newly_fetched),
            })
            continue

        # Claude returned findings array
        if isinstance(parsed, list):
            # Downgrade low-confidence criticals
            for f in parsed:
                if f.get("severity") == "critical" and f.get("confidence") != "high":
                    f["severity"] = "important"
                    f["title"] = f.get("title", "") + " (downgraded: unverified)"
            lookup_count = len(fetched_files)
            if lookup_count:
                print(f"  🔍 Used {lookup_count} extra file(s) for verification")
            return parsed

        print(f"  ⚠️  Unexpected response shape: {str(parsed)[:100]}")
        return []

    return []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def dedup_findings(findings: list[dict]) -> list[dict]:
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

SEVERITY_EMOJI = {"critical": "🔴", "important": "🟠", "minor": "🟢"}
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
            f"**Diff size:** {diff_lines} lines\n\n---\n\n✅ No issues found.\n"
        )

    sorted_findings = sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.get("severity", "minor"), 2),
            CONFIDENCE_ORDER.get(f.get("confidence", "low"), 2),
        ),
    )

    by_file: dict[str, list[dict]] = {}
    for f in sorted_findings:
        by_file.setdefault(f.get("file", "unknown"), []).append(f)

    critical  = sum(1 for f in sorted_findings if f.get("severity") == "critical")
    important = sum(1 for f in sorted_findings if f.get("severity") == "important")
    minor     = sum(1 for f in sorted_findings if f.get("severity") == "minor")

    lines = [
        "# Code Review",
        f"**Branch:** `{current_branch}` → `{base_branch}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Diff size:** {diff_lines} lines",
        f"**Findings:** 🔴 {critical} critical · 🟠 {important} important · 🟢 {minor} minor",
        "", "---", "",
    ]

    for file_path, file_findings in by_file.items():
        lines.append(f"## `{file_path}`")
        lines.append("")
        for f in file_findings:
            severity   = f.get("severity", "minor")
            confidence = f.get("confidence", "low")
            emoji      = SEVERITY_EMOJI.get(severity, "🟢")
            conf_note  = f" _(confidence: {confidence})_" if confidence != "high" else ""

            lines.append(f"### {emoji} {severity.upper()} — {f.get('title', 'Issue')}{conf_note}")
            lines.append("")
            if f.get("line"):
                lines.append(f"**Line:** {f['line']}")
                lines.append("")
            lines.append(f"**Problem:** {f.get('problem', '')}")
            lines.append("")
            if f.get("fix"):
                lines.append("**Fix:**")
                lines.append("```")
                lines.append(f.get("fix", ""))
                lines.append("```")
                lines.append("")
            if f.get("evidence"):
                lines.append(f"**Evidence:** _{f.get('evidence', '')}_")
                lines.append("")
            lines.append("---")
            lines.append("")

    lines += [
        "## Summary",
        "",
        f"🔴 **{critical} critical** — must fix before merge." if critical else "",
        f"🟠 **{important} important** — should fix before merge." if important else "",
        f"🟢 **{minor} minor** — consider fixing for code quality." if minor else "",
        "",
        "> ⚠️ Always verify CRITICAL findings manually.",
        "> Reviewer reads full file context + dependency files, but may miss product intent.",
    ]

    return "\n".join(l for l in lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def review_diff(
    base_branch: str = "main",
    db_path: str = "db/chroma",
    top_n: int = TOP_N_COMMENTS,
    project_config_path: str | None = None,
) -> None:
    openai_key    = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at '{db_path}'. Run embed.py first.")

    project_config = load_project_config(project_config_path)
    if project_config:
        print(f"Project config: {project_config.get('name', project_config_path)}")

    current_branch = get_current_branch()
    repo_root      = get_repo_root()

    print(f"Branch:    {current_branch}")
    print(f"Base:      {base_branch}")
    print(f"Repo root: {repo_root}")
    print(f"Diffing {base_branch}...{current_branch}\n")

    diff       = get_diff(base_branch, current_branch)
    diff_lines = diff.count("\n")
    file_diffs = parse_diff_by_file(diff)

    print(f"Diff size:     {diff_lines} lines")
    print(f"Changed files: {len(file_diffs)}")

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection    = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Knowledge base: {collection.count()} comments\n")

    openai_client = OpenAI(api_key=openai_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    SKIP_EXTENSIONS = {".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico"}
    project_context = build_project_context_section(project_config)
    all_findings: list[dict] = []

    for i, entry in enumerate(file_diffs, 1):
        file_path = entry["file"]
        file_diff = entry["diff"]
        hunks     = entry["hunks"]

        print(f"[{i}/{len(file_diffs)}] {file_path}")

        if any(file_path.endswith(ext) for ext in SKIP_EXTENSIONS):
            print("  Skipped (binary/lock)")
            continue

        file_context = read_file(file_path, repo_root, hunks)
        ctx_info = f"{file_context.count(chr(10))} lines" if file_context else "not available"
        print(f"  Context: {ctx_info}")

        similar = search_similar_comments(file_diff, openai_client, collection, top_n)
        print(f"  Past comments: {len(similar)}")

        # Auto-load always_include files from project config
        auto_files = get_always_include_files(file_path, project_config, repo_root)
        if auto_files:
            print(f"  Auto-included: {list(auto_files.keys())}")

        findings = agentic_review_file(
            file_path, file_diff, file_context, similar, repo_root, claude_client,
            project_context=project_context,
            project_config=project_config,
        )
        print(f"  Findings: {len(findings)}")
        all_findings.extend(findings)

    deduped = dedup_findings(all_findings)
    removed = len(all_findings) - len(deduped)
    if removed:
        print(f"\nDeduped {removed} duplicate(s)")

    markdown  = findings_to_markdown(deduped, current_branch, base_branch, diff_lines)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output    = f"review_{timestamp}.md"

    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n✅ Review saved to: {output}")
    print(f"   {len(deduped)} findings ({sum(1 for f in deduped if f.get('severity')=='critical')} critical)\n")


def main() -> None:
    # Read defaults from .env if set
    default_db      = os.getenv("AI_REVIEWER_DB", "db/chroma")
    default_project = os.getenv("AI_REVIEWER_PROJECT", None)

    parser = argparse.ArgumentParser(description="AI code reviewer with agentic file lookup")
    parser.add_argument("--base",    default="main", help="Base branch to diff against (default: main)")
    parser.add_argument("--db",      default=default_db, help="ChromaDB path (default: AI_REVIEWER_DB or db/chroma)")
    parser.add_argument("--top",     type=int, default=TOP_N_COMMENTS)
    parser.add_argument("--project", default=default_project, help="Path to project config json (default: AI_REVIEWER_PROJECT)")
    args = parser.parse_args()
    review_diff(base_branch=args.base, db_path=args.db, top_n=args.top, project_config_path=args.project)


if __name__ == "__main__":
    main()