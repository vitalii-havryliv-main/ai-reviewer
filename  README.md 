# AI Reviewer

AI code reviewer trained on your team's own PR comments.

## How it works

```
collect.py → reviews.json → embed.py → ChromaDB → review.py → AI review
```

1. **collect.py** — fetches review comments from GitHub PRs
2. **embed.py** — converts comments into vectors and stores in ChromaDB
3. **review.py** — takes current git diff, finds relevant comments, returns AI review

## Setup

```bash
git clone git@github.com:your-org/ai-reviewer.git
cd ai-reviewer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# add your tokens to .env
```

## .env

```
GITHUB_TOKEN=your_github_personal_access_token
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**GitHub token scopes required:** `repo` (for private repositories)

## Usage

**First time — collect and index comments:**
```bash
python collect.py --repo your-org/your-repo
python embed.py
```

**Every day — review your current changes:**
```bash
python review.py
```

**Update comments once a month:**
```bash
python collect.py --repo your-org/your-repo
python embed.py
```

## Project structure

```
ai-reviewer/
├── collect.py        # fetch PR comments from GitHub
├── embed.py          # build vector index from comments
├── review.py         # review current git diff
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

# generated locally, not in git:
├── data/
│   └── reviews.json
└── db/
    └── chroma/
```