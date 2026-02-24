# MMSAFE-Bench Architecture

## Design Philosophy

MMSAFE-Bench follows a modular pipeline architecture where each stage is independently testable and extensible. The framework is designed around three core principles:

1. **Provider-agnostic**: A unified `ModelProvider` ABC handles all model interactions, whether cloud APIs or local inference servers.
2. **Taxonomy-driven**: All safety evaluation is anchored to the MLCommons AILuminate v1.0 taxonomy (S1-S12) extended with 8 multi-modal hazard categories (X1-X8).
3. **Composable attacks and judges**: Attack strategies and safety judges are pluggable via registries, enabling new techniques without modifying core pipeline code.

## Pipeline Flow

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐    ┌──────────┐
│ Datasets │───▶│   Attacks    │───▶│  Providers   │───▶│   Judges   │───▶│ Metrics │───▶│ Reports  │
│  (JSONL) │    │ (transform)  │    │ (generate)   │    │ (evaluate) │    │ (ASR..) │    │ (HTML..) │
└──────────┘    └──────────────┘    └──────────────┘    └────────────┘    └─────────┘    └──────────┘
```

Each stage produces immutable data structures (`@dataclass(frozen=True)`) that flow through the pipeline.

## Key Interfaces

### ModelProvider (providers/base.py)
- `initialize()` → Set up API clients
- `generate(GenerationRequest)` → `GenerationResponse`
- `capabilities()` → `ProviderCapabilities`
- `health_check()` → `bool`
- `shutdown()` → Release resources

### AttackStrategy (attacks/base.py)
- `transform(GenerationRequest)` → `list[(GenerationRequest, AttackMetadata)]`

### SafetyJudge (judges/base.py)
- `evaluate(GenerationRequest, GenerationResponse)` → `SafetyVerdict`

## Module Map

| Module | Responsibility |
|--------|---------------|
| `config/` | Pydantic v2 configuration models with YAML loading |
| `taxonomy/` | Safety hazard taxonomy (S1-S12 + X1-X8) with severity and modality metadata |
| `datasets/` | JSONL dataset loading with schema validation |
| `attacks/` | 9 red-teaming strategies with variant support |
| `providers/` | 8 model provider adapters with rate limiting |
| `judges/` | 6 safety judges including weighted ensemble |
| `pipeline/` | Async evaluation runner with checkpoint/resume |
| `metrics/` | Statistical metrics with bootstrap confidence intervals |
| `reporting/` | HTML (Plotly charts), JSON, Markdown reports + leaderboard |
| `edge/` | Edge deployment simulation with 5 device profiles |
| `_internal/` | Shared utilities (logging, retry, hashing, concurrency) |

## Concurrency Model

The pipeline uses `asyncio` with a semaphore-based concurrency limiter (`bounded_gather`). Rate limiting is handled per-provider via a token-bucket algorithm. The execution config allows tuning concurrency, timeouts, and retry behavior.

## Testing Strategy

- **Unit tests**: Each module has corresponding tests in `tests/unit/`
- **Coverage gate**: 80% minimum enforced in CI
- **Seed dataset validation**: All shipped JSONL datasets are validated in tests
- **Deterministic stub provider**: Enables full pipeline testing without API keys
