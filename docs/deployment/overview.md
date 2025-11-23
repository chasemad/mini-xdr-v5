# Deployment Overview

This section covers the supported deployment paths for Mini-XDR. Choose the path appropriate for
your environment and keep the referenced artefacts updated when code changes.

## Targets

- **Local / Lab**: Use `docker-compose up -d` (or `scripts/start-all.sh`) with PostgreSQL, Redis, and
  local models. See [Local Quickstart](../getting-started/local-quickstart.md).
- **Kubernetes (optional)**: Manifests in `k8s/` for custom clusters. Requires external Postgres and
  secrets management. Validate manifests before production use.
- **Distributed (optional)**: Multi-node deployments with Kafka for event streaming, Redis for caching,
  and federated learning capabilities.

## Deployment Assets

| File/Folder | Purpose |
| --- | --- |
| `docker-compose.yml` | Local orchestration for Postgres, Redis, backend, frontend, optional T-Pot. |
| `.env.local` | Local environment defaults for backend/frontend. |
| `k8s/` | Kubernetes manifests for backend, frontend, and supporting services (validate before use). |
| `scripts/start-all.sh` | Local orchestration script—useful for smoke tests before releasing. |
| `docs/archived/aws/` | Legacy AWS deployment guides and artefacts (not part of the active stack). |

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
