# Deployment Overview

This section covers the supported deployment paths for Mini-XDR. Choose the path appropriate for
your environment and keep the referenced artefacts updated when code changes.

## Targets

- **Local / Lab**: Use `scripts/start-all.sh` with SQLite and local models. See
  [Local Quickstart](../getting-started/local-quickstart.md). Includes documentation enforcement system.
- **AWS**: EKS (Kubernetes), ECS/Fargate, or EC2-based deployments using artefacts in
  `infrastructure/aws/` and `k8s/`. Includes AWS Secrets Manager integration, CodeBuild CI/CD,
  and multi-tenant support.
- **Azure**: AKS + managed Postgres configuration documented under `deployment/azure/`.
  Supports multi-tenant deployments with Azure Key Vault integration.
- **Kubernetes (generic)**: Manifests in `k8s/` for custom clusters. Requires external Postgres and
  secrets management solution. Supports distributed deployments with Kafka/Redis.
- **Distributed**: Multi-node deployments with Kafka for event streaming, Redis for caching,
  and federated learning capabilities.

## Deployment Assets

| File/Folder | Purpose |
| --- | --- |
| `aws/` | Terraform and helper scripts for AWS networking, ECR, ECS tasks, and Secrets Manager. |
| `infrastructure/` | Shared IaC modules and automation (Terraform & shell). |
| `k8s/` | Kubernetes manifests for backend, frontend, and supporting services. |
| `buildspec-backend.yml`, `buildspec-frontend.yml` | AWS CodeBuild definitions (update if pipelines change). |
| `scripts/start-all.sh` | Local orchestration script—useful for smoke tests before releasing. |

## Release Checklist

1. Confirm documentation in this folder matches the versions in `backend/requirements.txt` and
   `frontend/package.json`.
2. Ensure container images are rebuilt (`docker build` definitions in `backend/Dockerfile`,
   `frontend/Dockerfile`).
3. Apply migrations (once implemented) against the target database.
4. Provision secrets in the relevant secret store and verify IAM roles.
5. Run smoke tests using `tests/test_system.sh` or the relevant CI suite.
6. Publish the final status in `change-control/release-notes.md`.

Refer to the platform-specific guides for detailed steps.
