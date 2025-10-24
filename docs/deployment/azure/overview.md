# Azure Deployment – Overview

Azure support is focused on running Mini-XDR inside AKS with managed Postgres and Key Vault. The
repository contains troubleshooting scripts under `scripts/azure/` and configuration guidance in
`ops/azure/`.

## Key Assets

| Path | Purpose |
| --- | --- |
| `scripts/azure/deploy-nginx-ingress.sh` | Installs NGINX ingress for AKS clusters used by Mini-XDR. |
| `scripts/azure/recreate-loadbalancer.sh` | Resets the AKS load balancer when stuck in `Pending`. |
| `scripts/azure/port-forward-demo.sh` | Local port forwarding helper for QuickStart validation. |
| `scripts/azure/DEMO_NOW.sh` | Convenience script for demo environments (AKS + frontend). |
| `scripts/azure-ml/` | ML workspace helpers (training jobs, dataset imports). |

## Prerequisites

- Azure CLI authenticated (`az login`).
- AKS cluster provisioned with worker node pools sized for FastAPI + Next.js pods.
- Azure Database for PostgreSQL or Cosmos DB (Postgres API) for persistent storage.
- Azure Key Vault for secrets.

## High-Level Steps

1. Provision resource group, AKS cluster, and database (Terraform or Azure CLI).
2. Populate Key Vault with the same secrets listed in
   [`getting-started/secrets-management.md`](../../getting-started/secrets-management.md).
3. Deploy backend and frontend manifests (see `k8s/` or Helm charts if available).
4. Configure ingress using `scripts/azure/deploy-nginx-ingress.sh` and update DNS.
5. Verify pods via `kubectl get pods`, check logs, and confirm API access through the ingress host.

See [operations](operations.md) and [troubleshooting](troubleshooting.md) for cluster-specific
procedures.
