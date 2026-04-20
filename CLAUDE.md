## Model Assignment Rules

- Architecture decisions and reviews: Use Opus
- Implementation tasks (new features, refactors): Use Sonnet  
- Simple edits, formatting, renaming: Use Haiku
- Security-sensitive changes: Always escalate to Opus for review

### Python Environment Detection 
At the start of every session, automatically detect and activate the bimap/
virtualenv for the current project

## Update CLAUDE.md for New Insights
- Run /revise-claude-md after a productive session to capture new insights.
- ask before running to confirm that the user wants to update the document.

## graphify
This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
