# Repository instructions

## Naming and public metadata

- Never use the terms `Codex` or `Claude`, in any capitalization, in branch
  names, commit subjects or bodies, pull request titles or descriptions,
  issue titles or descriptions, release names or notes, demo names, headings,
  summaries, generated metadata, or any other title or description.
- This instruction file is the sole exception needed to state and preserve the
  rule itself.
- Name work after the user-visible demo, outcome, behavior, or fix.
- Before creating a branch, commit, pull request, issue, or release, inspect
  its proposed name and descriptive text case-insensitively for the forbidden
  terms.

## Repository structure

- Keep demos under one shallow, outcome-based category at the repository root:
  `<category>/<outcome>/`.
- Keep every demo self-contained unless shared code is essential to understand
  or run it.
- Add every demo to `CATALOG.md` and the root `README.md`.
- Do not commit secrets, private endpoints, private inputs, logs, or production
  receipts.
- State local and hosted execution boundaries accurately.
