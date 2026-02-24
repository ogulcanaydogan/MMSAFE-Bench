"""Human evaluation interface — exports samples for annotation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mmsafe.judges.base import SafetyJudge, SafetyVerdict
from mmsafe.providers.base import GenerationRequest, GenerationResponse
from mmsafe.taxonomy.categories import ALL_MODALITIES, Modality


class HumanEvalJudge(SafetyJudge):
    """Exports evaluation samples to JSONL for human annotation.

    This judge always returns a neutral verdict and instead accumulates
    samples that can be exported for human reviewers.
    """

    def __init__(self, output_path: str = "artifacts/human_eval_queue.jsonl") -> None:
        self._output_path = Path(output_path)
        self._samples: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return "human_eval"

    @property
    def supported_modalities(self) -> tuple[Modality, ...]:
        return ALL_MODALITIES

    async def initialize(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._samples = []

    async def evaluate(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
    ) -> SafetyVerdict:
        start = time.monotonic()

        sample = {
            "prompt": request.prompt,
            "modality": request.modality.value,
            "model": request.model,
            "response_content": str(response.content) if response.content else "",
            "response_status": response.status.value,
            "refusal_detected": response.refusal_detected,
            "needs_human_review": True,
        }
        self._samples.append(sample)

        return SafetyVerdict(
            is_safe=True,
            confidence=0.0,
            explanation="Queued for human evaluation",
            judge_name=self.name,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def export(self) -> Path:
        """Export accumulated samples to JSONL file."""
        with open(self._output_path, "a") as f:
            for sample in self._samples:
                f.write(json.dumps(sample) + "\n")
        self._samples = []
        return self._output_path

    @property
    def pending_count(self) -> int:
        return len(self._samples)

    async def shutdown(self) -> None:
        if self._samples:
            self.export()
