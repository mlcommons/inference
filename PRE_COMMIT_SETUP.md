# Pre-Commit Hook Setup

This repository uses [pre-commit](https://pre-commit.com/) to enforce code formatting and secret scanning before each commit.

## Prerequisites

- Python 3.x
- `pip`
- For C/C++ formatting: `clang-format` (optional — pre-commit will install it via the mirror hook)

## Installation

1. **Install pre-commit:**

   ```bash
   pip install pre-commit
   ```

2. **Install the hooks into your local git repo:**

   ```bash
   pre-commit install
   ```

   This adds a `.git/hooks/pre-commit` script that runs automatically on every `git commit`.

## Hooks Configured

| Hook | What it does |
|------|-------------|
| `autopep8` | Auto-formats Python files to PEP 8 style (max line length 79). Excludes `tools/submission/power/power_checker.py`. |
| `clang-format` | Auto-formats C and C++ files using the repo's `.clang-format` style file. |
| `gitleaks` | Scans for hardcoded secrets, API keys, and credentials before they are committed. |

## Usage

Once installed, hooks run automatically on `git commit`. If a hook modifies files (e.g., autopep8 reformats a Python file), the commit is aborted so you can review the changes, then re-stage and commit:

```bash
git add <reformatted files>
git commit -m "your message"
```

## Running Hooks Manually

Run all hooks against all files:

```bash
pre-commit run --all-files
```

Run a specific hook:

```bash
pre-commit run autopep8
pre-commit run clang-format
pre-commit run gitleaks
```

## Skipping Hooks (Not Recommended)

In exceptional cases you can bypass hooks with:

```bash
git commit --no-verify -m "your message"
```

Use this sparingly — the hooks exist to keep the codebase consistent and secrets-free.

## Updating Hooks

To update hook versions to the latest revisions specified in `.pre-commit-config.yaml`:

```bash
pre-commit autoupdate
```
