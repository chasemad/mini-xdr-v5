# Azure Deployment – Operations

## Secrets & Configuration

- Use Azure Key Vault to store `API_KEY`, `JWT_SECRET_KEY`, and agent credentials. Reference them in
  your deployment manifests via Key Vault CSI driver or Azure App Config.
- Backend environment variables must match `backend/env.example`. Update the Kubernetes secret or
  deployment manifest when values change.

## AKS Cluster Checks

1. **Pods**: `kubectl get pods -n mini-xdr` – all pods should be `Running`.
2. **Ingress**: After running `scripts/azure/deploy-nginx-ingress.sh`, confirm the service has an
   external IP. Update DNS to point at the IP.
3. **ConfigMaps/Secrets**: Ensure the API key and database DSN are injected as env vars in the backend
   deployment (`kubectl describe deployment backend`).
4. **Database connectivity**: Backend logs (`kubectl logs deploy/backend`) should show successful
   connection to managed Postgres. Configure firewall rules to allow AKS outbound traffic.

## Frontend Notes

- Build the frontend with `npm run build` and serve via Next.js runtime container or static export +
  Azure Static Web Apps. Match `NEXT_PUBLIC_API_BASE` to the ingress hostname.

## Monitoring

- Enable Azure Monitor / Log Analytics. Forward FastAPI logs using `az monitor log-analytics workspace`.
- Set up alerts on pod restarts and ingress availability.

Record operational changes in `change-control/audit-log.md`.
