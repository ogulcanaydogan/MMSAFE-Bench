# Changelog

All notable changes to MMSAFE-Bench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-23

### Added
- Core evaluation pipeline with async execution and checkpoint/resume
- 20 hazard categories (MLCommons S1-S12 + custom X1-X8)
- 8 model providers: OpenAI, Anthropic, Google, Replicate, ElevenLabs, local vLLM, local Ollama, stub
- 9 attack strategies: passthrough, jailbreak, encoding, role-play, multi-turn, cross-modal, adversarial suffix, translation, composite
- 6 safety judges: keyword, LLM-as-judge, toxicity, NSFW classifier, composite ensemble, human evaluation export
- Safety metrics: ASR, RR, FRR, NSFW rate with bootstrap confidence intervals
- Reporting: HTML (Plotly charts), JSON, Markdown + model leaderboard
- Edge deployment simulation with 5 device profiles (DGX Spark, Jetson, RPi, Mac Studio, V100)
- 10 seed prompt datasets across text, image, video, audio, and cross-modal
- CLI with 8 commands: run, report, compare, leaderboard, providers, attacks, taxonomy, validate
- GPU-aware execution profiles (auto, small_gpu, a100)
- Docker support
- GitHub Actions CI (lint, test, smoke eval)
