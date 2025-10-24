# Azure Deployment – Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| AKS ingress stuck in `Pending` | Load balancer resource unavailable or stale public IP. | Run `scripts/azure/recreate-loadbalancer.sh` and verify subnet capacity. |
| Backend pods crashloop with `DATABASE_URL` errors | Secret missing or firewall blocking Postgres. | Reapply Kubernetes secret with correct DSN and whitelist AKS outbound IP in Azure Database for PostgreSQL. |
| 401 responses from API | `API_KEY` not set or mismatch between backend and UI. | Update both backend secret and frontend env (`NEXT_PUBLIC_API_KEY`). |
| Websocket disconnects | Ingress annotations missing for WebSockets. | Add `nginx.ingress.kubernetes.io/proxy-read-timeout` and upgrade to latest ingress controller; check `frontend/app/hooks/useIncidentRealtime.ts`. |
| Key Vault access denied | Pod identity/RBAC misconfigured. | Grant access using `az keyvault set-policy` for the managed identity used by the pod. |
