# Kubernetes & Infrastructure

## Kubernetes Manifests

- The `k8s/` directory contains manifests for backend, frontend, and supporting services. Review
  container images, resource requests, and Secrets references before applying to a cluster.
- **Multi-tenancy**: Configure tenant middleware settings and database isolation for multi-organization deployments.
- **Distributed Components**: For high-availability deployments, deploy Kafka and Redis services alongside the main application.
  Configure `backend/app/distributed/` components for event streaming and distributed caching.
- **ML Models**: Ensure the backend deployment mounts ML model artefacts from persistent storage or an init container
  that syncs `models/` from object storage (e.g., S3, Azure Blob Storage).
- **WebSocket Support**: Configure ingress according to your target provider (NGINX, ALB, Azure Application Gateway, etc.)
  and confirm WebSocket support for realtime updates and webhook integrations.
- **Secrets Management**: Set environment variables for `DATABASE_URL` (managed Postgres), `API_KEY`, `JWT_SECRET_KEY`,
  `ENCRYPTION_KEY`, and other required secrets via Kubernetes Secrets or external secret managers.
- **Documentation Enforcement**: The documentation validation system runs automatically in CI/CD pipelines.

## Terraform / IaC

- AWS infrastructure modules live in `infrastructure/aws/`. Includes VPC, EKS cluster, RDS Postgres,
  Secrets Manager, and ECR repository configurations. Update variables and outputs when resource
  names or requirements change.
- Azure infrastructure uses Terraform modules in `infrastructure/azure/` for AKS, Azure Database for
  PostgreSQL, Key Vault, and Container Registry.
- Keep Terraform state secured (S3 + DynamoDB lock for AWS, Azure Storage for Azure recommended).
  Document state file locations in `change-control/audit-log.md`.

## CI/CD Integration

- CodeBuild specs (`buildspec-backend.yml`, `buildspec-frontend.yml`) install dependencies and run
  builds. Adjust install commands when backend `requirements.txt` or frontend `package.json` change.
- For GitHub Actions or other CI, replicate the same steps: install Python deps, run tests, build
  Next.js, and package Docker images.

## Artefact Management

- ML models (`models/`) must be published to an artefact store (S3, Azure Blob) for cloud deployments.
  Update deployment manifests to mount or download the latest revisions.
- Container images should be versioned and pushed to your registry (ECR, ACR, GHCR). Tag releases in
  `change-control/release-notes.md`.

## Distributed Deployments

For high-availability and multi-node deployments:

1. **Kafka Integration**: Deploy Kafka cluster for event streaming between distributed nodes.
   Configure `backend/app/distributed/kafka_manager.py` with broker endpoints.
2. **Redis Clustering**: Deploy Redis cluster for distributed caching and session management.
   Configure `backend/app/distributed/redis_cluster.py` with cluster endpoints.
3. **Federated Learning**: Enable federated learning coordinator for privacy-preserving model training
   across distributed tenants.
4. **Load Balancing**: Configure load balancers to distribute traffic across multiple backend instances.
5. **Database**: Use managed PostgreSQL with read replicas for distributed read operations.

## Networking & Security

- Restrict ingress traffic to trusted IPs or load balancers. Backend relies on `x-api-key`, JWT, and HMAC;
  protect those credentials by enforcing TLS end-to-end.
- Configure network policies to restrict pod-to-pod communication based on tenant isolation requirements.
- Rotate TLS certificates and update ingress/load balancer resources accordingly.
- Enable mutual TLS (mTLS) for service-to-service communication in distributed deployments.
