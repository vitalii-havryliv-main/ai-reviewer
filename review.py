#!/usr/bin/env python3
"""
review.py — fast first-pass code reviewer ("raven" mode + cross-file leads)

Dumps broad candidate findings from the diff. It does NOT verify or rank by severity —
that is delegated to a separate counter-review step. It MAY grep the repo once per file to
surface cross-file inconsistency leads (the one thing the counter-review can't cheaply
reproduce). Cheap and fast on purpose.

Usage:
    python review.py
    python review.py --base 0.2.x
    python review.py --base main --project project.json

Setup:
    ANTHROPIC_API_KEY in your .env file.
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import re
import json
import time
import argparse
import subprocess
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
import anthropic
load_dotenv()

CLAUDE_MODEL    = "claude-sonnet-4-6"
MAX_RETRY_WAIT  = 120
MAX_LEAD_ROUNDS = 2     # at most one grep round, then a forced submit
GREP_MAX_LINES  = 40    # cap git-grep output per pattern
# Per-minute input-token budget (your API tier). Override via AI_REVIEWER_ITPM.
ITPM_LIMIT      = int(os.getenv("AI_REVIEWER_ITPM", "2000000"))

# Generated / non-reviewable files: skip entirely.
SKIP_EXTENSIONS = {".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
                   ".tsbuildinfo", ".map", ".snap", ".min.js", ".min.css"}
SKIP_FILENAMES  = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                   "composer.lock", "Gemfile.lock", "poetry.lock", "Cargo.lock"}
MAX_DIFF_TOKENS = 30000  # files this large are almost certainly generated


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


def grep_repo(pattern: str, repo_root: str, max_lines: int = GREP_MAX_LINES) -> str:
    """git grep (tracked + untracked) so the model can surface cross-file leads cheaply."""
    try:
        r = subprocess.run(
            ["git", "grep", "-n", "-I", "--no-color", "--untracked", "-e", pattern],
            cwd=repo_root, capture_output=True, text=True, timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        return f"(grep failed for '{pattern}': {e})"
    out = r.stdout.strip()
    if not out:
        return f"(no matches for: {pattern})"
    rows = out.split("\n")
    capped = "\n".join(rows[:max_lines])
    extra = f"\n… (+{len(rows) - max_lines} more)" if len(rows) > max_lines else ""
    return f"matches for '{pattern}':\n{capped}{extra}"


# ---------------------------------------------------------------------------
# Project config (optional context, cached in the system prompt)
# ---------------------------------------------------------------------------

def load_project_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    config_path = os.path.expanduser(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Project config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_project_context_section(project_config: dict) -> str:
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
        if langs:
            parts.append(f"Languages: {langs}")
        if frameworks:
            parts.append(f"Frameworks: {frameworks}")
        for lib in stack.get("key_libs", []):
            parts.append(f"  - {lib}")
    for key, label in (
        ("structure", "Project structure"),
        ("intentional_patterns", "INTENTIONAL PATTERNS (do NOT flag these)"),
        ("review_rules", "PROJECT REVIEW RULES"),
        ("additional_context", "ADDITIONAL PROJECT CONTEXT"),
    ):
        val = project_config.get(key, "")
        if val:
            parts.append(f"\n{label}:\n{val.strip()}")
    if not parts:
        return ""
    return "--- PROJECT CONTEXT ---\n" + "\n".join(parts) + "\n--- END PROJECT CONTEXT ---"


# ---------------------------------------------------------------------------
# Diff parsing (per-file, diff text only)
# ---------------------------------------------------------------------------

def parse_diff_by_file(diff: str) -> list[dict]:
    files = []
    current: dict | None = None
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            if current:
                files.append(current)
            match = re.search(r" b/(.+)$", line)
            file_path = match.group(1) if match else "unknown"
            current = {"file": file_path, "diff": line + "\n"}
        elif current is not None:
            current["diff"] += line + "\n"
    if current:
        files.append(current)
    return files


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TokenRateLimiter:
    def __init__(self, limit: int, window: float = 60.0, safety: float = 0.95):
        self.budget = max(1, int(limit * safety))
        self.window = window
        self.events: deque[tuple[float, int]] = deque()

    def _used(self, now: float) -> int:
        while self.events and now - self.events[0][0] > self.window:
            self.events.popleft()
        return sum(tok for _, tok in self.events)

    def acquire(self, est_tokens: int) -> None:
        while True:
            now = time.monotonic()
            used = self._used(now)
            if not self.events or used + est_tokens <= self.budget:
                return
            sleep_for = self.window - (now - self.events[0][0]) + 0.5
            if sleep_for <= 0:
                continue
            print(f"  ⏳ Pacing: ~{used} tok in last 60s, waiting {sleep_for:.0f}s...")
            time.sleep(sleep_for)

    def record(self, tokens: int) -> None:
        self.events.append((time.monotonic(), tokens))


_RATE_LIMITER = TokenRateLimiter(ITPM_LIMIT)


def _estimate_input_tokens(system, messages) -> int:
    chars = 0
    for block in (system or []):
        chars += len(block.get("text", ""))
    for msg in messages:
        content = msg.get("content", "")
        chars += len(content) if isinstance(content, str) else len(str(content))
    return chars // 3


def create_message_with_retry(claude_client, max_attempts: int = 8, **kwargs):
    est = _estimate_input_tokens(kwargs.get("system"), kwargs.get("messages", []))
    for attempt in range(1, max_attempts + 1):
        _RATE_LIMITER.acquire(est)
        try:
            response = claude_client.messages.create(**kwargs)
            usage = response.usage
            # Cache reads are cheap and limited separately — don't count them.
            billed = usage.input_tokens + getattr(usage, "cache_creation_input_tokens", 0)
            _RATE_LIMITER.record(billed)
            return response
        except anthropic.RateLimitError as err:
            if attempt == max_attempts:
                raise
            retry_after = None
            try:
                retry_after = float(err.response.headers.get("retry-after"))
            except (AttributeError, TypeError, ValueError):
                pass
            wait = retry_after if retry_after is not None else min(60, 2 ** attempt)
            if wait > MAX_RETRY_WAIT:
                raise RuntimeError(
                    f"Rate limited for {wait:.0f}s — requests too large for the tier."
                ) from err
            print(f"  ⏳ Rate limited; waiting {wait:.0f}s (attempt {attempt}/{max_attempts})...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Review (single-shot, diff-only)
# ---------------------------------------------------------------------------

FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "file":       {"type": "string"},
        "line":       {"type": "string"},
        "title":      {"type": "string"},
        "problem":    {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "crossfile":  {"type": "boolean",
                       "description": "true if found via grep as a cross-file inconsistency lead"},
    },
    "required": ["file", "title", "problem"],
}

GREP_TOOL = {
    "name": "grep",
    "description": (
        "Search the repo (git grep) to surface CROSS-FILE inconsistency leads — e.g. compare a "
        "changed symbol against an existing sibling hook/util/type. Use at most once. Do not try "
        "to confirm anything; just gather enough to phrase a lead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patterns": {"type": "array", "items": {"type": "string"}},
            "reason":   {"type": "string"},
        },
        "required": ["patterns"],
    },
}

SUBMIT_TOOL = {
    "name": "submit_findings",
    "description": "Submit all candidate findings. Call exactly once when done.",
    "input_schema": {
        "type": "object",
        "properties": {"findings": {"type": "array", "items": FINDING_SCHEMA}},
        "required": ["findings"],
    },
}


def build_system_prompt(project_context: str = "") -> str:
    project_section = ("\n" + project_context + "\n") if project_context else ""
    return f"""You are a fast first-pass code reviewer.
{project_section}
Look mainly at the diff below and list EVERY plausible issue from the changed lines —
bugs, risky patterns, missing error handling, type issues, dead code. Over-reporting is
fine; a separate step verifies everything. Be fast and broad.

CROSS-FILE LEADS (the most valuable output): when a changed symbol resembles an existing
sibling (hook/util/type/component), you MAY call `grep` ONCE to find it and surface
inconsistencies as leads — set crossfile=true and phrase the title as "cross-file: check
X vs Y". Do NOT try to confirm a lead; just surface it for the verification step.

Do NOT flag these (predictable noise):
- import sources (which module a symbol is imported from) or whether code is in/out of
  project scope — you don't know the conventions or the roadmap;
- deleted/renamed/removed exports as "may break consumers" — a separate step verifies refs.
- NEVER claim a file or symbol "does not exist" / "is missing". git grep is NOT
  authoritative for absence (untracked files, path guesses, generated code). If grep
  returns nothing, drop the point — do not report it.
- If the SAME inconsistency spans several files (e.g. one rename), emit ONE lead that
  lists the affected files — do not repeat it per file.

Keep each `problem` to ONE short sentence. Set `confidence` honestly. After at most one
grep, call submit_findings with all candidates."""


def review_file(file_path: str, file_diff: str, repo_root: str, claude_client,
                project_context: str = "") -> list[dict]:
    system = [{
        "type": "text",
        "text": build_system_prompt(project_context),
        "cache_control": {"type": "ephemeral"},
    }]
    messages = [{
        "role": "user",
        "content": f"Review this diff for `{file_path}`. Report every plausible issue.\n\n{file_diff}",
    }]
    tools = [GREP_TOOL, SUBMIT_TOOL]

    for round_num in range(1, MAX_LEAD_ROUNDS + 1):
        final = round_num == MAX_LEAD_ROUNDS
        tool_choice = {"type": "tool", "name": "submit_findings"} if final else {"type": "any"}
        response = create_message_with_retry(
            claude_client, model=CLAUDE_MODEL, max_tokens=4000,
            system=system, messages=messages, tools=tools, tool_choice=tool_choice,
        )
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            return []

        submit = next((b for b in tool_uses if b.name == "submit_findings"), None)
        if submit:
            findings = submit.input.get("findings", []) if isinstance(submit.input, dict) else []
            return findings if isinstance(findings, list) else []

        # grep round — surface leads, feed results back, then force submit next round.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for b in tool_uses:
            if b.name == "grep":
                patterns = (b.input or {}).get("patterns", [])[:8]
                if patterns:
                    print(f"  🔎 grep: {', '.join(patterns)}")
                out = "\n\n".join(grep_repo(p, repo_root) for p in patterns) or "(no patterns)"
            else:
                out = f"Unknown tool: {b.name}"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})

    return []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _norm_title(title: str) -> str:
    t = re.sub(r"\(downgraded[^)]*\)", "", title or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", "", t)  # strip backticks/punctuation
    return re.sub(r"\s+", " ", t).strip()[:60]


def dedup_findings(findings: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for f in findings:
        key = f"{f.get('file', '')}::{_norm_title(f.get('title', ''))}"
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _problem(f: dict) -> str:
    p = re.sub(r"\s+", " ", f.get("problem", "")).strip()
    return p[:200].rstrip() + "…" if len(p) > 200 else p


def _conf_tag(f: dict) -> str:
    conf = f.get("confidence", "low")
    return "" if conf == "high" else f" _({conf})_"


def findings_to_markdown(findings, current_branch, base_branch, diff_lines) -> str:
    header = (
        f"# Code Review (first-pass, unverified)\n"
        f"**Branch:** `{current_branch}` → `{base_branch}`\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Diff size:** {diff_lines} lines\n"
    )
    if not findings:
        return header + "\n✅ No candidate issues found.\n"

    leads = [f for f in findings if f.get("crossfile")]
    rest  = sorted(
        (f for f in findings if not f.get("crossfile")),
        key=lambda f: CONFIDENCE_ORDER.get(f.get("confidence", "low"), 2),
    )

    lines = [
        header.rstrip("\n"),
        f"**Candidates:** {len(findings)}  ·  🔗 {len(leads)} cross-file leads  "
        f"—  unverified first-pass, run counter-review",
        "",
    ]

    # Cross-file leads first — the part the counter-review can't cheaply reproduce.
    if leads:
        lines.append("## 🔗 Cross-file leads (check these first)")
        for f in sorted(leads, key=lambda f: CONFIDENCE_ORDER.get(f.get("confidence", "low"), 2)):
            loc = f":{f['line']}" if f.get("line") else ""
            lines.append(f"- **{f.get('title', 'Issue')}**{_conf_tag(f)} "
                         f"— `{f.get('file', '?')}{loc}` — {_problem(f)}")
        lines.append("")

    by_file: dict[str, list[dict]] = {}
    for f in rest:
        by_file.setdefault(f.get("file", "unknown"), []).append(f)
    for file_path, file_findings in by_file.items():
        lines.append(f"## `{file_path}`")
        for f in file_findings:
            loc = f" L{f['line']}" if f.get("line") else ""
            lines.append(f"- **{f.get('title', 'Issue')}**{_conf_tag(f)}{loc} — {_problem(f)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def review_diff(base_branch: str = "main", project_config_path: str | None = None) -> None:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env")

    project_config = load_project_config(project_config_path)
    if project_config:
        print(f"Project config: {project_config.get('name', project_config_path)}")

    current_branch = get_current_branch()
    repo_root      = get_repo_root()
    print(f"Branch: {current_branch}")
    print(f"Base:   {base_branch}")
    print(f"Diffing {base_branch}...{current_branch}\n")

    diff       = get_diff(base_branch, current_branch)
    diff_lines = diff.count("\n")
    file_diffs = parse_diff_by_file(diff)
    print(f"Diff size:     {diff_lines} lines")
    print(f"Changed files: {len(file_diffs)}\n")

    claude_client   = anthropic.Anthropic(api_key=anthropic_key, max_retries=4)
    project_context = build_project_context_section(project_config)
    all_findings: list[dict] = []

    for i, entry in enumerate(file_diffs, 1):
        file_path, file_diff = entry["file"], entry["diff"]
        print(f"[{i}/{len(file_diffs)}] {file_path}")

        basename = os.path.basename(file_path)
        if any(file_path.endswith(ext) for ext in SKIP_EXTENSIONS) or basename in SKIP_FILENAMES:
            print("  Skipped (generated/binary/lock)")
            continue
        if len(file_diff) // 3 > MAX_DIFF_TOKENS:
            print("  Skipped (huge diff — looks generated)")
            continue

        findings = review_file(file_path, file_diff, repo_root, claude_client, project_context)
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

    print(f"\n✅ Review saved to: {output}  ({len(deduped)} candidate findings)\n")


def main() -> None:
    default_project = os.getenv("AI_REVIEWER_PROJECT", None)
    parser = argparse.ArgumentParser(description="Fast first-pass (diff-only) code reviewer")
    parser.add_argument("--base",    default="main", help="Base branch to diff against")
    parser.add_argument("--project", default=default_project, help="Path to project config json")
    args = parser.parse_args()
    review_diff(base_branch=args.base, project_config_path=args.project)


if __name__ == "__main__":
    main()
