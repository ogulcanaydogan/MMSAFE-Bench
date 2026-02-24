# Security Policy

## Purpose

MMSAFE-Bench is a **defensive safety evaluation tool** designed to help AI developers and researchers identify safety vulnerabilities in generative AI models. It performs controlled red-teaming in test environments to improve model safety.

## Responsible Use

This tool is intended for:
- Safety researchers evaluating model robustness
- AI developers testing safety guardrails before deployment
- Organizations conducting compliance audits

This tool is **not** intended for:
- Generating harmful content for distribution
- Bypassing safety mechanisms in production systems
- Any use that violates applicable laws or regulations

## Reporting Vulnerabilities

If you discover a security vulnerability in MMSAFE-Bench itself, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email: security@yapay.dev
3. Include: description, reproduction steps, potential impact

We aim to acknowledge reports within 48 hours and provide a fix within 7 days for critical issues.

## API Key Security

- API keys are loaded from environment variables via pydantic-settings
- Keys are never logged or included in reports
- The `.env.example` file documents required variables without values
- `detect-secrets` is configured in pre-commit hooks

## Dataset Safety

Prompt datasets contain adversarial content by design. They are clearly labeled and should be handled with care:
- Datasets are stored in `datasets/` with clear naming
- Benign datasets are flagged with `is_benign: true`
- Raw model outputs are excluded from reports by default (`include_raw_samples: false`)
