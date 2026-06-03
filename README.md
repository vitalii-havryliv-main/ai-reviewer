# AI Reviewer

AI code reviewer trained on your team's own PR comments.

## How it works

```
collect.py → reviews.json → embed.py → ChromaDB → review.py → AI review
```

1. **collect.py** — fetches review comments from GitHub PRs
2. **embed.py** — converts comments into vectors and stores in ChromaDB
3. **review.py** — diffs your branch against base, finds relevant past comments, returns AI review

---

## First time setup

```bash
git clone git@github.com:your-org/ai-reviewer.git
cd ai-reviewer

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# fill in your tokens and paths in .env
```

### .env

```
GITHUB_TOKEN=your_github_personal_access_token
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key

# set these once — no need to pass --db and --project on every run
AI_REVIEWER_DB=/absolute/path/to/ai-reviewer/db/chroma
AI_REVIEWER_PROJECT=/absolute/path/to/ai-reviewer/project.json
```

**GitHub token scopes required:** `repo` (for private repositories)

### Build the knowledge base

```bash
mkdir data
python collect.py --repo your-org/your-repo --output data/reviews.json
python embed.py --input data/reviews.json --db db/chroma
```

### Set up project config

```bash
cp project.example.json project.json
```

Open `project.example.json` — it contains a prompt. Give it to Claude or Cursor and ask it to analyze your codebase and fill in `project.json`. Review the output and adjust if needed.

### Set up alias (optional but recommended)

Add to your `~/.zshrc`:

```bash
alias review="source /absolute/path/to/ai-reviewer/venv/bin/activate && python /absolute/path/to/ai-reviewer/review.py"
```

Then run `source ~/.zshrc`.

---

## Run a review

Go to your feature branch and run:

```bash
cd ~/your-project
review --base your-base-branch
```

The review is saved as `review_YYYY-MM-DD_HH-MM.md` in the current directory.

---

## Quick restart (already set up, just need to run again)

```bash
cd ~/your-project
git checkout your-feature-branch
review --base your-base-branch
```

That's it — venv activates via the alias, paths are read from `.env`.

---

## Update knowledge base (once a month)

```bash
cd ~/ai-reviewer
source venv/bin/activate
python collect.py --repo your-org/your-repo --output data/reviews.json
python embed.py --input data/reviews.json --db db/chroma
```

---

## Project structure

```
ai-reviewer/
├── collect.py            # fetch PR comments from GitHub
├── embed.py              # build vector index from comments
├── review.py             # review current git diff
├── project.example.json  # template — give to AI to generate project.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

# generated locally, not in git:
├── .env
├── project.json          # your filled project config
├── data/
│   └── reviews.json
└── db/
    └── chroma/
```
