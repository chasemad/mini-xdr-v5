import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.nlp_workflow_parser import parse_workflow_from_natural_language


@pytest.mark.asyncio
async def test_malware_host_isolation_detected():
    intent, explanation = await parse_workflow_from_natural_language(
        "Create a malware response workflow for incident #5 with host isolation and memory dumping",
        incident_id=5
    )

    action_types = {action['action_type'] for action in intent.actions}

    assert 'isolate_host_advanced' in action_types
    assert 'memory_dump_collection' in action_types
    assert intent.priority in ('high', 'critical')
    assert intent.approval_required is True
    assert 'workflow steps' in explanation.lower()


@pytest.mark.asyncio
async def test_credential_stuffing_identity_actions_detected():
    intent, _ = await parse_workflow_from_natural_language(
        "Implement credential stuffing defense including password reset and MFA enforcement",
        incident_id=3
    )

    action_types = {action['action_type'] for action in intent.actions}

    assert 'reset_passwords' in action_types
    assert 'enforce_mfa' in action_types
    assert intent.priority in ('high', 'critical')
    assert intent.approval_required is True


@pytest.mark.asyncio
async def test_ddos_rate_limiting_and_traffic_analysis_detected():
    intent, _ = await parse_workflow_from_natural_language(
        "Set up DDoS protection for the affected servers with rate limiting and traffic analysis",
        incident_id=7
    )

    action_types = {action['action_type'] for action in intent.actions}

    assert 'api_rate_limiting' in action_types
    assert 'capture_network_traffic' in action_types
    assert intent.priority in ('high', 'critical')
    assert intent.approval_required is True


@pytest.mark.asyncio
async def test_phishing_response_email_actions_detected():
    intent, _ = await parse_workflow_from_natural_language(
        "Respond to the phishing campaign by quarantining the emails and blocking the sender domain",
        incident_id=9
    )

    action_types = {action['action_type'] for action in intent.actions}

    assert 'quarantine_email' in action_types
    assert 'block_sender' in action_types
    assert intent.priority in ('medium', 'high', 'critical')


@pytest.mark.asyncio
async def test_insider_threat_requires_identity_actions():
    intent, _ = await parse_workflow_from_natural_language(
        "Investigate insider threat activity and disable the compromised user account",
        incident_id=11
    )

    action_types = {action['action_type'] for action in intent.actions}

    assert 'investigate_behavior' in action_types
    assert 'disable_user_account' in action_types
    assert intent.priority in ('high', 'critical')
    assert intent.approval_required is True
