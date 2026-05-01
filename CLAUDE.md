## Model Assignment Rules
- Architecture decisions and reviews: Use Opus
- Implementation tasks (new features, refactors): Use Sonnet  
- Simple edits, formatting, renaming: Use Haiku
- Security-sensitive changes: Always escalate to Opus for review

## Python Environment
- Virtualenv: `bimap/` — activate with `bimap\Scripts\activate` (Windows) or `source bimap/bin/activate` (Unix)
- Dependencies: `requirements.txt` (no pyproject.toml; project is not pip-installed)
- At the start of every session, automatically detect and activate the bimap/ virtualenv

## Running the Pipeline
```bash
# Single seed
python pipeline.py --config configs/phase0_mvp.yaml --seed 42

# Multiple seeds (embeddings computed once, eval repeated per seed)
python pipeline.py --config configs/phase0_mvp.yaml --seeds 42 123 2024
```

## Update CLAUDE.md for New Insights
- Run /revise-claude-md after a productive session to capture new insights.
- Ask before running to confirm that the user wants to update the document.

## graphify
This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

-each line of your code will be reviewd by codex, so please make sure to write clean and well-commented code. If you have any questions about the codebase or architecture, refer to the graphify knowledge graph for insights before asking.
