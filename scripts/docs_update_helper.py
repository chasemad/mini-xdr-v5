#!/usr/bin/env python3
"""
Documentation Update Helper for Mini-XDR

This script helps developers identify what documentation needs to be updated
when they make code changes. It provides specific guidance and templates.

Usage:
    python scripts/docs_update_helper.py [--staged] [--interactive]

Options:
    --staged        Check only staged changes
    --interactive   Interactive mode to help update docs
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


class DocsUpdateHelper:
    """Helper class to guide documentation updates"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def get_changed_files(self, staged: bool = False) -> List[str]:
        """Get list of changed files"""
        try:
            if staged:
                cmd = ["git", "diff", "--cached", "--name-only"]
            else:
                cmd = ["git", "diff", "--name-only"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        except Exception as e:
            print(f"Error getting changed files: {e}")
            return []

    def analyze_changes(self, staged: bool = False) -> Dict[str, List[str]]:
        """Analyze changes and suggest documentation updates"""
        changed_files = self.get_changed_files(staged)
        suggestions = {}

        # API changes
        api_files = [
            f
            for f in changed_files
            if f.startswith("backend/app/") and f.endswith(".py")
        ]
        if api_files:
            api_suggestions = self._analyze_api_changes(api_files)
            if api_suggestions:
                suggestions["API Documentation"] = api_suggestions

        # Model changes
        model_files = [f for f in changed_files if "models.py" in f or "db.py" in f]
        if model_files:
            suggestions["Architecture Documentation"] = [
                "Update docs/architecture/system-overview.md with new database models",
                "Update docs/architecture/data-flows.md with new data flows",
                "Check docs/api/reference.md for any new data structures",
            ]

        # Configuration changes
        config_files = [
            f
            for f in changed_files
            if any(x in f for x in ["config.py", "env.example", "main.py"])
        ]
        if config_files:
            suggestions["Configuration Documentation"] = [
                "Update docs/getting-started/environment-config.md with new variables",
                "Update docs/getting-started/secrets-management.md if secrets changed",
            ]

        # UI changes
        ui_files = [
            f
            for f in changed_files
            if f.startswith("frontend/") and f.endswith((".tsx", ".ts"))
        ]
        if ui_files:
            suggestions["UI Documentation"] = [
                "Update docs/ui/dashboard-guide.md for dashboard changes",
                "Update docs/ui/automation-designer.md for workflow designer changes",
            ]

        # Infrastructure changes
        infra_files = [
            f
            for f in changed_files
            if any(x in f for x in ["infrastructure/", "k8s/", "scripts/"])
        ]
        if infra_files:
            suggestions["Deployment Documentation"] = [
                "Update docs/deployment/overview.md with new deployment options",
                "Update docs/deployment/aws/overview.md for AWS changes",
                "Update docs/deployment/azure/overview.md for Azure changes",
                "Update docs/deployment/kubernetes-and-infra.md for k8s/infra changes",
            ]

        # Security changes
        security_files = [
            f
            for f in changed_files
            if any(x in f for x in ["security.py", "auth.py", "policies/"])
        ]
        if security_files:
            suggestions["Security Documentation"] = [
                "Update docs/security-compliance/security-hardening.md",
                "Update docs/security-compliance/auth-and-access.md",
                "Update docs/change-control/audit-log.md for security changes",
            ]

        # ML changes
        ml_files = [
            f
            for f in changed_files
            if any(
                x in f
                for x in [
                    "ml_engine.py",
                    "enhanced_threat_detector.py",
                    "models/",
                    "ml-training/",
                ]
            )
        ]
        if ml_files:
            suggestions["ML Documentation"] = [
                "Update docs/ml/training-guide.md for training changes",
                "Update docs/ml/model-ops-runbook.md for deployment changes",
                "Update docs/ml/data-sources.md for new data sources",
            ]

        return suggestions

    def _analyze_api_changes(self, api_files: List[str]) -> List[str]:
        """Analyze API files for new endpoints"""
        suggestions = []

        for file_path in api_files:
            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Look for FastAPI route decorators
                routes = re.findall(
                    r'@app\.(get|post|put|delete|patch)\([\'"]([^\'"]*)[\'"]', content
                )
                if routes:
                    suggestions.append(f"New API endpoints found in {file_path}:")
                    for method, path in routes:
                        suggestions.append(f"  - {method.upper()} {path}")
                    suggestions.append(
                        "→ Update docs/api/reference.md with these endpoints"
                    )

                # Look for Pydantic models
                models = re.findall(r"class (\w+)\(.*BaseModel\)", content)
                if models:
                    suggestions.append(
                        f"New API models found in {file_path}: {', '.join(models)}"
                    )
                    suggestions.append(
                        "→ Update docs/api/reference.md with request/response schemas"
                    )

            except Exception as e:
                suggestions.append(f"Error analyzing {file_path}: {e}")

        if suggestions:
            suggestions.insert(0, "API Changes Detected:")
            suggestions.append(
                "→ Update docs/api/workflows-and-integrations.md if workflows changed"
            )

        return suggestions

    def generate_update_plan(self, suggestions: Dict[str, List[str]]) -> str:
        """Generate a formatted update plan"""
        if not suggestions:
            return "✅ No documentation updates required for current changes."

        plan = ["📋 Documentation Update Plan", "=" * 40, ""]

        for category, items in suggestions.items():
            plan.append(f"🔧 {category}")
            plan.append("-" * (len(category) + 3))
            plan.extend(items)
            plan.append("")

        plan.append("📖 Contributing Guidelines:")
        plan.append("1. Keep statements factual—mirror the behaviour in code")
        plan.append("2. Document production defaults AND local overrides")
        plan.append("3. Update docs in the same pull request as code changes")
        plan.append("4. Link directly to files (e.g., backend/app/main.py)")
        plan.append("5. Use ASCII text and wrap at ~100 characters")
        plan.append("")

        plan.append("🛠️  Quick Commands:")
        plan.append("# Stage your documentation changes")
        plan.append("git add docs/")
        plan.append("")
        plan.append("# Validate your documentation updates")
        plan.append("python scripts/validate_docs_update.py --staged")
        plan.append("")
        plan.append("# View documentation changes")
        plan.append("git diff --cached docs/")

        return "\n".join(plan)

    def interactive_mode(self, staged: bool = False):
        """Interactive mode to help update documentation"""
        suggestions = self.analyze_changes(staged)

        if not suggestions:
            print("✅ No documentation updates required!")
            return

        print(self.generate_update_plan(suggestions))
        print("\n" + "=" * 60)

        # Interactive help
        print("\n💡 Interactive Documentation Helper")
        print("-" * 40)

        response = (
            input(
                "\nWould you like help opening the relevant documentation files? (y/n): "
            )
            .lower()
            .strip()
        )

        if response == "y":
            self._open_relevant_docs(suggestions)

        response = (
            input("\nWould you like to see examples of documentation updates? (y/n): ")
            .lower()
            .strip()
        )

        if response == "y":
            self._show_examples(suggestions)

    def _open_relevant_docs(self, suggestions: Dict[str, List[str]]):
        """Open relevant documentation files"""
        docs_to_open = set()

        # Extract file paths from suggestions
        for items in suggestions.values():
            for item in items:
                # Look for file paths in suggestions
                matches = re.findall(r"docs/[\w\-/]+\.md", item)
                docs_to_open.update(matches)

        if docs_to_open:
            print(f"\n📂 Opening {len(docs_to_open)} documentation files...")
            for doc in sorted(docs_to_open):
                doc_path = self.project_root / doc
                if doc_path.exists():
                    print(f"  • {doc}")
                    # Try to open with default editor
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(["open", str(doc_path)])
                        elif sys.platform == "linux":
                            subprocess.run(["xdg-open", str(doc_path)])
                        elif sys.platform == "win32":
                            subprocess.run(["start", str(doc_path)], shell=True)
                    except Exception as e:
                        print(f"    Could not auto-open {doc}: {e}")
                else:
                    print(f"  ⚠️  {doc} does not exist yet")

    def _show_examples(self, suggestions: Dict[str, List[str]]):
        """Show examples of documentation updates"""
        print("\n📝 Documentation Update Examples")
        print("-" * 40)

        if "API Documentation" in suggestions:
            print(
                """
🔧 API Documentation Example:

## Authentication (`/api/auth/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/auth/verify-2fa` | Verify 2FA token for enhanced security. |

Add this to: docs/api/reference.md
"""
            )

        if "Architecture Documentation" in suggestions:
            print(
                """
🔧 Architecture Documentation Example:

## Persistence

- SQLAlchemy async engine configured in `backend/app/db.py` with SQLite default, Postgres support via
  AsyncPG DSNs.
- **NEW:** Added audit logging table (`audit_logs`) in `backend/app/models.py` to track administrative
  actions for compliance.

Add this to: docs/architecture/system-overview.md
"""
            )

        if "Configuration Documentation" in suggestions:
            print(
                """
🔧 Configuration Documentation Example:

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_2FA` | `false` | Enables two-factor authentication for all users. |

Add this to: docs/getting-started/environment-config.md
"""
            )

        print("\n💡 Pro Tip: Use the existing documentation style and formatting.")
        print("   Check similar sections in the docs for consistency.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Help identify required documentation updates"
    )
    parser.add_argument(
        "--staged", action="store_true", help="Check only staged changes"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode with help"
    )

    args = parser.parse_args()

    helper = DocsUpdateHelper()

    if args.interactive:
        helper.interactive_mode(staged=args.staged)
    else:
        suggestions = helper.analyze_changes(staged=args.staged)
        print(helper.generate_update_plan(suggestions))


if __name__ == "__main__":
    main()
