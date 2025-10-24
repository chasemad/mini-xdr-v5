#!/usr/bin/env python3
"""
Documentation Validation Script for Mini-XDR

This script validates that code changes are properly documented according to
enterprise standards. It should be run as a pre-commit hook or CI check.

Usage:
    python scripts/validate_docs_update.py [--staged] [--commit-hash <hash>]

Options:
    --staged        Check only staged changes (for pre-commit)
    --commit-hash   Check changes in specific commit
    --help         Show this help message
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class DocValidationRule:
    """Represents a documentation validation rule"""

    name: str
    description: str
    file_patterns: List[str]
    required_docs: List[str]
    validation_func: callable = None


class DocValidator:
    """Validates documentation completeness for code changes"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.rules = self._define_rules()

    def _define_rules(self) -> List[DocValidationRule]:
        """Define all documentation validation rules"""
        return [
            DocValidationRule(
                name="api_changes",
                description="API endpoint changes require docs/api/reference.md updates",
                file_patterns=["backend/app/**/*.py"],
                required_docs=["docs/api/reference.md"],
                validation_func=self._validate_api_changes,
            ),
            DocValidationRule(
                name="model_changes",
                description="Database model changes require architecture documentation updates",
                file_patterns=["backend/app/models.py", "backend/app/db.py"],
                required_docs=[
                    "docs/architecture/system-overview.md",
                    "docs/architecture/data-flows.md",
                ],
                validation_func=self._validate_model_changes,
            ),
            DocValidationRule(
                name="config_changes",
                description="Configuration changes require environment config updates",
                file_patterns=[
                    "backend/app/config.py",
                    "backend/env.example",
                    "backend/app/main.py",
                ],
                required_docs=["docs/getting-started/environment-config.md"],
                validation_func=self._validate_config_changes,
            ),
            DocValidationRule(
                name="ui_changes",
                description="Frontend component changes require UI documentation updates",
                file_patterns=["frontend/app/**/*.tsx", "frontend/app/**/*.ts"],
                required_docs=[
                    "docs/ui/dashboard-guide.md",
                    "docs/ui/automation-designer.md",
                ],
                validation_func=self._validate_ui_changes,
            ),
            DocValidationRule(
                name="deployment_changes",
                description="Infrastructure changes require deployment documentation updates",
                file_patterns=[
                    "infrastructure/**/*.tf",
                    "infrastructure/**/*.yaml",
                    "k8s/**/*.yaml",
                    "scripts/**/*.sh",
                ],
                required_docs=[
                    "docs/deployment/overview.md",
                    "docs/deployment/aws/overview.md",
                    "docs/deployment/azure/overview.md",
                ],
                validation_func=self._validate_deployment_changes,
            ),
            DocValidationRule(
                name="security_changes",
                description="Security changes require security compliance documentation updates",
                file_patterns=[
                    "backend/app/security.py",
                    "backend/app/auth.py",
                    "policies/**/*.yaml",
                    "backend/app/policy_engine.py",
                ],
                required_docs=[
                    "docs/security-compliance/security-hardening.md",
                    "docs/security-compliance/auth-and-access.md",
                ],
                validation_func=self._validate_security_changes,
            ),
            DocValidationRule(
                name="ml_changes",
                description="ML/model changes require ML documentation updates",
                file_patterns=[
                    "backend/app/ml_engine.py",
                    "backend/app/enhanced_threat_detector.py",
                    "models/**/*",
                    "scripts/ml-training/**/*.py",
                ],
                required_docs=[
                    "docs/ml/training-guide.md",
                    "docs/ml/model-ops-runbook.md",
                    "docs/ml/data-sources.md",
                ],
                validation_func=self._validate_ml_changes,
            ),
            DocValidationRule(
                name="workflow_changes",
                description="Response workflow changes require workflow documentation updates",
                file_patterns=[
                    "backend/app/advanced_response_engine.py",
                    "backend/app/response_optimizer.py",
                    "backend/app/workflow_*.py",
                ],
                required_docs=[
                    "docs/api/workflows-and-integrations.md",
                    "docs/architecture/data-flows.md",
                ],
                validation_func=self._validate_workflow_changes,
            ),
        ]

    def get_changed_files(
        self, staged: bool = False, commit_hash: str = None
    ) -> List[str]:
        """Get list of changed files"""
        try:
            if commit_hash:
                cmd = [
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit_hash,
                ]
            elif staged:
                cmd = ["git", "diff", "--cached", "--name-only"]
            else:
                cmd = ["git", "diff", "--name-only"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode != 0:
                print(f"Error getting changed files: {result.stderr}")
                return []

            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        except Exception as e:
            print(f"Error getting changed files: {e}")
            return []

    def check_file_matches_pattern(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file matches any of the patterns"""
        from fnmatch import fnmatch

        return any(fnmatch(file_path, pattern) for pattern in patterns)

    def get_modified_docs(
        self, staged: bool = False, commit_hash: str = None
    ) -> Set[str]:
        """Get set of documentation files that were modified"""
        changed_files = self.get_changed_files(staged, commit_hash)
        return {f for f in changed_files if f.startswith("docs/")}

    def validate_changes(
        self, staged: bool = False, commit_hash: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that code changes have corresponding documentation updates

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        changed_files = self.get_changed_files(staged, commit_hash)
        modified_docs = self.get_modified_docs(staged, commit_hash)

        issues = []
        triggered_rules = []

        # Check each rule
        for rule in self.rules:
            relevant_changes = [
                f
                for f in changed_files
                if self.check_file_matches_pattern(f, rule.file_patterns)
            ]

            if not relevant_changes:
                continue

            triggered_rules.append(rule)

            # Check if required docs were modified
            missing_docs = []
            for required_doc in rule.required_docs:
                if required_doc not in modified_docs:
                    missing_docs.append(required_doc)

            if missing_docs:
                issues.append(f"Rule '{rule.name}': {rule.description}")
                issues.append(f"  Changed files: {', '.join(relevant_changes)}")
                issues.append(
                    f"  Missing documentation updates: {', '.join(missing_docs)}"
                )
                issues.append("")

                # Run specific validation if available
                if rule.validation_func:
                    try:
                        specific_issues = rule.validation_func(
                            relevant_changes, modified_docs
                        )
                        if specific_issues:
                            issues.extend(specific_issues)
                            issues.append("")
                    except Exception as e:
                        issues.append(f"  Error in specific validation: {e}")
                        issues.append("")

        # Summary
        if issues:
            issues.insert(
                0,
                f"❌ Documentation validation failed! {len(triggered_rules)} rules triggered.",
            )
            issues.insert(1, "")
            return False, issues
        else:
            return True, ["✅ All documentation validation checks passed!"]

    def _validate_api_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for API changes"""
        issues = []

        # Check if new endpoints were added
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for FastAPI route decorators
                new_routes = re.findall(
                    r'@app\.(get|post|put|delete|patch)\([\'"]([^\'"]*)[\'"]', content
                )
                if new_routes and "docs/api/reference.md" not in modified_docs:
                    issues.append(
                        "  ⚠️  New API endpoints detected but docs/api/reference.md not updated"
                    )
                    for method, path in new_routes:
                        issues.append(f"    - {method.upper()} {path}")
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_model_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for model changes"""
        issues = []

        # Check for new model classes or field changes
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for SQLAlchemy model definitions
                if "class" in content and "Base" in content:
                    if not any(
                        doc in modified_docs
                        for doc in [
                            "docs/architecture/system-overview.md",
                            "docs/architecture/data-flows.md",
                        ]
                    ):
                        issues.append(
                            "  ⚠️  Model changes detected but architecture docs not updated"
                        )
                        break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_config_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for configuration changes"""
        issues = []

        # Check for new environment variables
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for settings or environment variables
                if "os.getenv" in content or "settings." in content:
                    if (
                        "docs/getting-started/environment-config.md"
                        not in modified_docs
                    ):
                        issues.append(
                            "  ⚠️  Configuration changes detected but environment config docs not updated"
                        )
                        break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_ui_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for UI changes"""
        issues = []

        # Check for new components or major UI changes
        ui_docs = ["docs/ui/dashboard-guide.md", "docs/ui/automation-designer.md"]
        if any(doc in modified_docs for doc in ui_docs):
            return []  # Already updated

        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for component definitions or major UI patterns
                if any(
                    keyword in content
                    for keyword in [
                        "export const",
                        "export default function",
                        "export function",
                    ]
                ):
                    issues.append(
                        "  ⚠️  UI component changes detected but UI docs not updated"
                    )
                    break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_deployment_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for deployment changes"""
        issues = []

        deployment_docs = [
            "docs/deployment/overview.md",
            "docs/deployment/aws/overview.md",
            "docs/deployment/azure/overview.md",
        ]

        if any(doc in modified_docs for doc in deployment_docs):
            return []  # Already updated

        # Check if infrastructure files changed significantly
        infra_patterns = ["infrastructure/", "k8s/", "scripts/"]
        infra_changes = [
            f for f in changed_files if any(pattern in f for pattern in infra_patterns)
        ]

        if infra_changes:
            issues.append(
                "  ⚠️  Infrastructure changes detected but deployment docs not updated"
            )
            issues.append("    Consider updating: docs/deployment/overview.md")

        return issues

    def _validate_security_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for security changes"""
        issues = []

        security_docs = [
            "docs/security-compliance/security-hardening.md",
            "docs/security-compliance/auth-and-access.md",
        ]

        if any(doc in modified_docs for doc in security_docs):
            return []  # Already updated

        # Check for security-related changes
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for security-related keywords
                security_keywords = [
                    "auth",
                    "security",
                    "encrypt",
                    "decrypt",
                    "token",
                    "password",
                    "secret",
                ]
                if any(keyword in content.lower() for keyword in security_keywords):
                    issues.append(
                        "  ⚠️  Security-related changes detected but security docs not updated"
                    )
                    break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_ml_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for ML changes"""
        issues = []

        ml_docs = [
            "docs/ml/training-guide.md",
            "docs/ml/model-ops-runbook.md",
            "docs/ml/data-sources.md",
        ]

        if any(doc in modified_docs for doc in ml_docs):
            return []  # Already updated

        # Check for ML-related changes
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for ML-related keywords
                ml_keywords = [
                    "model",
                    "train",
                    "predict",
                    "ml_engine",
                    "detector",
                    "dataset",
                ]
                if any(keyword in content.lower() for keyword in ml_keywords):
                    issues.append(
                        "  ⚠️  ML-related changes detected but ML docs not updated"
                    )
                    break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues

    def _validate_workflow_changes(
        self, changed_files: List[str], modified_docs: Set[str]
    ) -> List[str]:
        """Specific validation for workflow changes"""
        issues = []

        workflow_docs = [
            "docs/api/workflows-and-integrations.md",
            "docs/architecture/data-flows.md",
        ]

        if any(doc in modified_docs for doc in workflow_docs):
            return []  # Already updated

        # Check for workflow-related changes
        for file in changed_files:
            if not os.path.exists(file):
                continue

            try:
                with open(file, "r") as f:
                    content = f.read()

                # Look for workflow-related keywords
                workflow_keywords = [
                    "workflow",
                    "response",
                    "action",
                    "automation",
                    "orchestrate",
                ]
                if any(keyword in content.lower() for keyword in workflow_keywords):
                    issues.append(
                        "  ⚠️  Workflow-related changes detected but workflow docs not updated"
                    )
                    break
            except Exception as e:
                issues.append(f"  Error reading {file}: {e}")

        return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate documentation completeness for code changes"
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check only staged changes (for pre-commit)",
    )
    parser.add_argument("--commit-hash", help="Check changes in specific commit")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on validation failures",
    )

    args = parser.parse_args()

    validator = DocValidator()
    is_valid, issues = validator.validate_changes(
        staged=args.staged, commit_hash=args.commit_hash
    )

    for issue in issues:
        print(issue)

    if not is_valid and args.strict:
        sys.exit(1)

    return is_valid


if __name__ == "__main__":
    main()
