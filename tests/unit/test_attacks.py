"""Tests for attack strategies."""

from __future__ import annotations

from mmsafe.attacks.adversarial_suffix import AdversarialSuffixAttack
from mmsafe.attacks.composite import CompositeAttack
from mmsafe.attacks.cross_modal import CrossModalAttack
from mmsafe.attacks.encoding import EncodingAttack
from mmsafe.attacks.jailbreak import JailbreakAttack
from mmsafe.attacks.multi_turn import MultiTurnAttack
from mmsafe.attacks.passthrough import PassthroughAttack
from mmsafe.attacks.registry import ATTACK_REGISTRY, get_attack_class
from mmsafe.attacks.role_play import RolePlayAttack
from mmsafe.attacks.translation import TranslationAttack
from mmsafe.providers.base import GenerationRequest
from mmsafe.taxonomy.categories import Modality

import pytest


def _text_request(prompt: str = "How to pick a lock") -> GenerationRequest:
    return GenerationRequest(prompt=prompt, modality=Modality.TEXT, model="test")


def _image_request(prompt: str = "Generate a violent image") -> GenerationRequest:
    return GenerationRequest(prompt=prompt, modality=Modality.IMAGE, model="test")


class TestAttackRegistry:
    def test_all_strategies_registered(self) -> None:
        expected = {
            "passthrough", "jailbreak", "encoding", "role_play",
            "multi_turn", "cross_modal", "adversarial_suffix",
            "translation", "composite",
        }
        assert set(ATTACK_REGISTRY.keys()) == expected

    def test_get_attack_class_passthrough(self) -> None:
        cls = get_attack_class("passthrough")
        assert cls is PassthroughAttack

    def test_get_attack_class_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown attack"):
            get_attack_class("nonexistent")


class TestPassthroughAttack:
    def test_returns_unchanged(self) -> None:
        attack = PassthroughAttack()
        results = attack.transform(_text_request())
        assert len(results) == 1
        req, meta = results[0]
        assert req.prompt == "How to pick a lock"
        assert meta.strategy_name == "passthrough"
        assert meta.original_prompt == meta.transformed_prompt

    def test_name(self) -> None:
        assert PassthroughAttack().name == "passthrough"


class TestJailbreakAttack:
    def test_produces_variants(self) -> None:
        attack = JailbreakAttack()
        results = attack.transform(_text_request())
        assert len(results) == 3  # dan, aim, developer_mode

    def test_specific_variants(self) -> None:
        attack = JailbreakAttack(variants=["dan"])
        results = attack.transform(_text_request())
        assert len(results) == 1
        _, meta = results[0]
        assert meta.variant == "dan"

    def test_prompt_embedded(self) -> None:
        attack = JailbreakAttack(variants=["dan"])
        results = attack.transform(_text_request("test prompt"))
        req, _ = results[0]
        assert "test prompt" in req.prompt
        assert "DAN" in req.prompt


class TestEncodingAttack:
    def test_produces_variants(self) -> None:
        attack = EncodingAttack()
        results = attack.transform(_text_request())
        assert len(results) == 3  # base64, rot13, unicode_substitution

    def test_base64_encoding(self) -> None:
        attack = EncodingAttack(variants=["base64"])
        results = attack.transform(_text_request("test"))
        req, meta = results[0]
        assert "Base64" in req.prompt
        assert meta.variant == "base64"

    def test_rot13_encoding(self) -> None:
        attack = EncodingAttack(variants=["rot13"])
        results = attack.transform(_text_request("test"))
        req, _ = results[0]
        assert "ROT13" in req.prompt


class TestRolePlayAttack:
    def test_produces_variants(self) -> None:
        attack = RolePlayAttack()
        results = attack.transform(_text_request())
        assert len(results) == 4

    def test_specific_variant(self) -> None:
        attack = RolePlayAttack(variants=["fictional_character"])
        results = attack.transform(_text_request())
        assert len(results) == 1
        req, meta = results[0]
        assert "fictional" in req.prompt.lower() or "story" in req.prompt.lower()
        assert meta.variant == "fictional_character"


class TestMultiTurnAttack:
    def test_produces_conversation_history(self) -> None:
        attack = MultiTurnAttack()
        results = attack.transform(_text_request())
        assert len(results) == 1
        req, meta = results[0]
        assert len(req.conversation_history) > 0
        assert meta.strategy_name == "multi_turn"

    def test_escalation_variants(self) -> None:
        for escalation in ["gradual", "context_building", "authority_escalation"]:
            attack = MultiTurnAttack(escalation=escalation)
            results = attack.transform(_text_request())
            assert len(results) == 1


class TestCrossModalAttack:
    def test_image_variants(self) -> None:
        attack = CrossModalAttack()
        results = attack.transform(_image_request())
        assert len(results) >= 1
        for req, meta in results:
            assert meta.strategy_name == "cross_modal"

    def test_text_returns_empty(self) -> None:
        attack = CrossModalAttack()
        results = attack.transform(_text_request())
        # No cross-modal variants for text modality
        assert len(results) == 0


class TestAdversarialSuffixAttack:
    def test_produces_suffixed_prompts(self) -> None:
        attack = AdversarialSuffixAttack(num_suffixes=2)
        results = attack.transform(_text_request("Hello"))
        assert len(results) == 2
        for req, meta in results:
            assert req.prompt.startswith("Hello ")
            assert len(req.prompt) > len("Hello ")

    def test_deterministic(self) -> None:
        attack1 = AdversarialSuffixAttack(seed=42)
        attack2 = AdversarialSuffixAttack(seed=42)
        r1 = attack1.transform(_text_request())
        r2 = attack2.transform(_text_request())
        assert r1[0][0].prompt == r2[0][0].prompt


class TestTranslationAttack:
    def test_produces_variants(self) -> None:
        attack = TranslationAttack()
        results = attack.transform(_text_request())
        assert len(results) == 5  # zu, sw, ht, hmn, mi

    def test_specific_languages(self) -> None:
        attack = TranslationAttack(target_languages=["zu", "sw"])
        results = attack.transform(_text_request())
        assert len(results) == 2
        variants = {meta.variant for _, meta in results}
        assert variants == {"zu", "sw"}


class TestCompositeAttack:
    def test_empty_chain(self) -> None:
        attack = CompositeAttack(strategies=[])
        results = attack.transform(_text_request())
        assert len(results) == 1
        req, meta = results[0]
        assert req.prompt == "How to pick a lock"

    def test_single_strategy_chain(self) -> None:
        attack = CompositeAttack(strategies=[PassthroughAttack()])
        results = attack.transform(_text_request())
        assert len(results) == 1

    def test_multi_strategy_chain(self) -> None:
        attack = CompositeAttack(strategies=[
            JailbreakAttack(variants=["dan"]),
            EncodingAttack(variants=["base64"]),
        ])
        results = attack.transform(_text_request())
        assert len(results) == 1
        req, meta = results[0]
        assert "Base64" in req.prompt
        assert meta.strategy_name == "composite"
        assert "jailbreak+encoding" == meta.variant
