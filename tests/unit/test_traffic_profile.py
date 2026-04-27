"""Unit tests for system_prompt_tokens population in TrafficProfileGenerator."""

import pytest
from unittest.mock import MagicMock

from planner.specification.traffic_profile import TrafficProfileGenerator
from planner.shared.schemas import DeploymentIntent


def _make_intent(use_case: str, user_count: int = 100) -> DeploymentIntent:
    return DeploymentIntent(
        use_case=use_case,
        experience_class="conversational",
        user_count=user_count,
        latency_priority="medium",
    )


def _make_mock_slo_repo(has_template: bool = True):
    repo = MagicMock()
    if has_template:
        template = MagicMock()
        template.prompt_tokens = 512
        template.output_tokens = 256
        template.ttft_p95_target_ms = 300
        template.itl_p95_target_ms = 30
        template.e2e_p95_target_ms = 25000
        repo.get_template.return_value = template
    else:
        repo.get_template.return_value = None
    return repo


@pytest.mark.unit
def test_system_prompt_tokens_chatbot_conversational():
    """chatbot_conversational use case should populate system_prompt_tokens=400."""
    repo = _make_mock_slo_repo(has_template=True)
    gen = TrafficProfileGenerator(slo_repo=repo)
    profile = gen.generate_profile(_make_intent("chatbot_conversational"))
    assert profile.system_prompt_tokens == 400


@pytest.mark.unit
def test_system_prompt_tokens_unmapped_use_case():
    """Use case not in defaults dict should produce system_prompt_tokens=0."""
    repo = _make_mock_slo_repo(has_template=True)
    gen = TrafficProfileGenerator(slo_repo=repo)
    # 'translation' is a valid use case but not in SYSTEM_PROMPT_TOKEN_DEFAULTS
    profile = gen.generate_profile(_make_intent("translation"))
    assert profile.system_prompt_tokens == 0
