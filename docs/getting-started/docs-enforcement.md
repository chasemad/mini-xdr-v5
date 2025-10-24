# Documentation Enforcement Guide

This guide explains Mini-XDR's automated documentation enforcement system that ensures all code changes are properly documented like a professional enterprise organization.

## Overview

The documentation enforcement system consists of:

1. **Pre-commit hooks** - Automatic validation before commits
2. **CI/CD validation** - GitHub Actions checks on pull requests
3. **Interactive helpers** - Tools to guide documentation updates
4. **Validation rules** - Comprehensive rules covering all system components

## Quick Start

### 1. Install Pre-commit Hooks

```bash
# Install pre-commit (one-time setup)
pip install pre-commit
pre-commit install

# Optional: Run on all files to check current state
pre-commit run --all-files
```

### 2. Make Code Changes

Develop your features/bug fixes as usual:

```bash
# Make your changes
git add .

# Pre-commit will automatically validate documentation
git commit -m "feat: add new API endpoint"
```

If documentation validation fails, you'll see output like:

```
❌ Documentation validation failed! 2 rules triggered.

Rule 'api_changes': API endpoint changes require docs/api/reference.md updates
  Changed files: backend/app/new_endpoint.py
  Missing documentation updates: docs/api/reference.md
```

## Interactive Documentation Helper

When you need help identifying what to document:

```bash
# Analyze current changes and get specific guidance
python scripts/docs_update_helper.py --staged

# Interactive mode with file opening and examples
python scripts/docs_update_helper.py --staged --interactive
```

Example output:

```
🔧 API Documentation
--------------------
API Changes Detected:
New API endpoints found in backend/app/alerts.py:
  - GET /api/alerts/summary
  - POST /api/alerts/{id}/escalate
→ Update docs/api/reference.md with these endpoints

📖 Contributing Guidelines:
1. Keep statements factual—mirror the behaviour in code
2. Document production defaults AND local overrides
3. Update docs in the same pull request as code changes
4. Link directly to files (e.g., backend/app/main.py)
5. Use ASCII text and wrap at ~100 characters
```

## Validation Rules

### API Changes
- **Files**: `backend/app/**/*.py`
- **Required Docs**: `docs/api/reference.md`
- **Validates**: New FastAPI route decorators, Pydantic models

### Database/Model Changes
- **Files**: `backend/app/models.py`, `backend/app/db.py`
- **Required Docs**: `docs/architecture/system-overview.md`, `docs/architecture/data-flows.md`
- **Validates**: New SQLAlchemy models, schema changes

### Configuration Changes
- **Files**: `backend/app/config.py`, `backend/env.example`
- **Required Docs**: `docs/getting-started/environment-config.md`
- **Validates**: New environment variables, settings changes

### UI Changes
- **Files**: `frontend/**/*.tsx`, `frontend/**/*.ts`
- **Required Docs**: `docs/ui/dashboard-guide.md`, `docs/ui/automation-designer.md`
- **Validates**: New components, major UI updates

### Infrastructure Changes
- **Files**: `infrastructure/`, `k8s/`, `scripts/*.sh`
- **Required Docs**: `docs/deployment/overview.md`, AWS/Azure deployment docs
- **Validates**: New infrastructure components

### Security Changes
- **Files**: `backend/app/security.py`, `backend/app/auth.py`, `policies/`
- **Required Docs**: `docs/security-compliance/`
- **Validates**: Security hardening, authentication changes

### ML Changes
- **Files**: `backend/app/ml_engine.py`, `models/`, `scripts/ml-training/`
- **Required Docs**: `docs/ml/`
- **Validates**: Model updates, training changes

## Documentation Standards

### API Reference Updates

When adding new endpoints, add to `docs/api/reference.md`:

```markdown
## Alerts (`/api/alerts/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/alerts/summary` | Get alert summary statistics. |
| POST | `/api/alerts/{id}/escalate` | Escalate alert to next severity level. |
```

### Architecture Updates

When changing data flows, update `docs/architecture/data-flows.md`:

```markdown
## Response Automation

- **Auto Containment**: `settings.auto_contain` toggles immediate actions in
  `backend/app/responder.py` (e.g., `block_ip`, `isolate_host`).
- **NEW**: Alert escalation workflows in `backend/app/alert_manager.py` trigger
  notifications to SOC operators when severity thresholds are exceeded.
```

### Configuration Updates

When adding environment variables, update `docs/getting-started/environment-config.md`:

```markdown
| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_ALERT_ESCALATION` | `true` | Enables automatic alert escalation based on severity rules. |
```

## CI/CD Integration

### Pull Request Validation

GitHub Actions automatically validates documentation on PRs:

```yaml
# .github/workflows/docs-validation.yml
- name: Validate documentation completeness
  run: |
    python scripts/validate_docs_update.py --commit-hash ${{ github.sha }} --strict
```

### Status Checks

PRs must pass documentation validation before merging. The status check will show:
- ✅ All documentation validation checks passed
- ❌ Documentation validation failed

## Troubleshooting

### Pre-commit Hook Issues

```bash
# Skip hooks for urgent commits (not recommended)
git commit --no-verify -m "urgent: fix critical bug"

# Run specific hook manually
pre-commit run validate-docs-update --all-files
```

### Validation False Positives

If validation incorrectly flags changes:

1. Check if your documentation updates are staged: `git add docs/`
2. Run validation manually: `python scripts/validate_docs_update.py --staged`
3. If still failing, check the specific error messages for guidance

### Documentation Structure Changes

When restructuring documentation:

1. Update the validation rules in `scripts/validate_docs_update.py`
2. Update this guide with new file locations
3. Test the changes with `python scripts/validate_docs_update.py --staged`

## Enterprise Benefits

This system ensures:

1. **Consistency**: All changes follow the same documentation standards
2. **Completeness**: No feature goes undocumented
3. **Accuracy**: Documentation stays synchronized with code
4. **Compliance**: Audit trail of all system changes
5. **Onboarding**: New team members have current documentation
6. **Maintenance**: Easier to maintain and update systems with good docs

## Contributing to the Enforcement System

When the validation rules need updates:

1. Edit `scripts/validate_docs_update.py` for new rules
2. Update `.cursorrules` for Cursor integration
3. Test with `python scripts/validate_docs_update.py --staged`
4. Update this guide with any new processes

Remember: Good documentation is as important as good code in enterprise systems.
