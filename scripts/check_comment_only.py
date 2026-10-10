#!/usr/bin/env python3
"""
Prove that edits to Python files changed only comments and docstrings (the T7.4 documentation pass).

For every given file (default: every modified ``.py`` file in the working tree), the syntax tree
of the working copy is compared with the one at ``git HEAD`` after removing docstrings from both.
Comments never reach the syntax tree, so equal trees mean the code itself did not change.

    python3 scripts/check_comment_only.py                 # all modified .py files
    python3 scripts/check_comment_only.py backend/app.py  # just these

Exit 0 when every file matches, 1 (listing the files) when any code changed.

Concept: a *syntax tree* (abstract syntax tree, AST) is what Python builds from source before it
runs anything: one node per statement, expression and name. Formatting, blank lines and comments
are not in it, so two files with the same tree run the same code. ``ast.dump`` turns a tree into a
plain string, which makes "did the code change?" a single string comparison.

Docstrings *are* in the tree (a docstring is just a string expression as the first statement of a
module, class or function), which is why they are stripped from both sides before comparing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from typing import List


def without_docstrings(source: str) -> str:
    """``ast.dump`` of the source with every module/class/function docstring removed."""
    tree = ast.parse(source)
    # ast.walk visits every node in the tree, nested functions and classes included.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            # A docstring is an expression statement whose value is a string constant.
            # Any other first statement (an assignment, a call) is real code and stays.
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                # Keep the body non-empty: a function whose only statement was its docstring needs a pass.
                node.body = node.body[1:] or [ast.Pass()]
    # include_attributes=False drops line and column numbers, so a comment that shifts
    # later code down a few lines does not count as a change.
    return ast.dump(tree, include_attributes=False)


def modified_python_files() -> List[str]:
    """List the ``.py`` files that differ from ``git HEAD`` in the working tree."""
    # check=True raises if git fails (for example outside a repository) instead of
    # silently reporting no files and passing.
    out = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "*.py"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def main(argv: List[str]) -> int:
    """Compare each file with its ``HEAD`` version and return the process exit code."""
    files = argv or modified_python_files()
    changed = []
    for path in files:
        # ``git show HEAD:path`` prints the committed version without touching the working tree.
        before = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True)
        if before.returncode != 0:
            print(f"skip {path}: not in HEAD (a new file)")
            continue
        with open(path) as handle:
            after = handle.read()
        if without_docstrings(before.stdout) != without_docstrings(after):
            changed.append(path)
    for path in changed:
        print(f"CODE CHANGED: {path}")
    print(f"{len(files) - len(changed)} of {len(files)} files are comment/docstring-only changes")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
