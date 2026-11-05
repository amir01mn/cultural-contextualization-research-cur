"""Enforce the subpackage layering documented in docs/ARCHITECTURE.md.

A subpackage may import only from itself and the packages listed for it here.
If a change needs a new edge, update docs/ARCHITECTURE.md and this table
together rather than silently widening the dependency graph.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

ALLOWED = {
    "core": set(),
    "llm": {"core"},
    "loaders": {"core"},
    "construction": {"core", "llm"},
    "retrieval": {"core"},
    "evaluation": {"core", "llm", "loaders", "retrieval"},
    "legacy": {"core", "llm", "retrieval"},
}


def _imported_packages(path: Path, own: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 1:
            found.add(own)
        elif node.level == 2 and node.module:
            found.add(node.module.split(".")[0])
        elif node.level == 0 and node.module and node.module.startswith("src."):
            found.add(node.module.split(".")[1])
    return found


def test_every_module_lives_in_a_known_subpackage():
    stray = [p.name for p in SRC.glob("*.py") if p.name != "__init__.py"]
    assert not stray, f"modules at src/ top level: {stray}; place them in a subpackage"
    unknown = {p.name for p in SRC.iterdir()
               if p.is_dir() and p.name != "__pycache__"} - set(ALLOWED)
    assert not unknown, f"undocumented subpackages: {unknown}"


def test_subpackages_respect_the_dependency_order():
    violations = []
    for pkg, allowed in ALLOWED.items():
        for path in sorted((SRC / pkg).glob("*.py")):
            extra = _imported_packages(path, pkg) - allowed - {pkg}
            if extra:
                violations.append(f"{pkg}/{path.name} imports {sorted(extra)}")
    assert not violations, "\n".join(violations)


def test_only_legacy_code_imports_legacy():
    offenders = []
    roots = [SRC, SRC.parent / "scripts", SRC.parent / "tests"]
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(SRC.parent).as_posix()
            if rel.startswith(("src/legacy/", "scripts/legacy/", "tests/test_legacy_")):
                continue
            if path.name == Path(__file__).name:
                continue
            text = path.read_text()
            if "src.legacy" in text or "from ..legacy" in text:
                offenders.append(rel)
    assert not offenders, f"non-legacy code imports src.legacy: {offenders}"
