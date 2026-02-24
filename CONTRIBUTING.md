# Contributing to MMSAFE-Bench

## Development Setup

```bash
git clone https://github.com/ogulcanaydogan/MMSAFE-Bench.git
cd MMSAFE-Bench
uv sync --extra dev --extra viz
pre-commit install
```

## Running Tests

```bash
make test          # Full test suite with coverage
make test-unit     # Unit tests only
make lint          # Ruff + mypy
make fmt           # Auto-format
```

## Adding a New Provider

1. Create `mmsafe/providers/your_provider.py` implementing `ModelProvider`
2. Register it in `mmsafe/providers/registry.py`
3. Add tests in `tests/unit/test_providers.py`

## Adding a New Attack Strategy

1. Create `mmsafe/attacks/your_attack.py` implementing `AttackStrategy`
2. Register it in `mmsafe/attacks/registry.py`
3. Add tests in `tests/unit/test_attacks.py`

## Adding a New Judge

1. Create `mmsafe/judges/your_judge.py` implementing `SafetyJudge`
2. Register it in `mmsafe/judges/registry.py`
3. Add tests in `tests/unit/test_judges.py`

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Tests
- `refactor:` — Code restructure
- `chore:` — Maintenance

## Code Style

- Python 3.12+
- Ruff for linting and formatting
- mypy strict mode for type checking
- `from __future__ import annotations` in all files
