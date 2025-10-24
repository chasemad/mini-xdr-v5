# Monitoring & Alerts

Mini-XDR currently exposes telemetry through API endpoints, structured logging, and optional metrics.

## Backend Logging

- Uvicorn/FastAPI logs appear on stdout and include scheduler activity, ML loading, incident actions,
  agent orchestration, and distributed component status (`logging` configured in `backend/app/main.py`).
- Structured logging includes tenant context, agent operations, workflow executions, and federated learning events.
- When running on servers, route logs to CloudWatch (AWS), Azure Monitor, or ELK stack. For local debugging,
  tail logs with `uvicorn --log-level info` or inspect `backend/backend.log` if file logging is enabled.
- Distributed deployments generate coordinated logs across Kafka streams and Redis operations.

## Telemetry API

- `/api/telemetry/status` returns per-organization asset counts, enrolled agents, incident totals,
  last event timestamp, and tenant-specific metrics.
- `/api/orchestrator/status` provides agent orchestration health and active workflow counts.
- `/api/ml/status` and `/api/ml/online-learning/status` report ML model performance and training status.
- `/api/federated/status` shows federated learning coordinator and participant health.
- `/api/distributed/status` and `/api/distributed/health` provide distributed system monitoring.
- Frontend widgets consume these endpoints to display comprehensive system health on dashboards.

## Scheduler Health

- Background jobs include `process_scheduled_unblocks`, `background_retrain_ml_models`, federated learning
  coordination, concept drift monitoring, and online learning adaptation.
- APScheduler manages all background tasks; monitor job execution status and queue depths.
- Alert if jobs fail repeatedly, exceed expected runtime, or queue backlogs exceed thresholds.
- Distributed deployments require coordination across multiple scheduler instances.

## Metrics

- The repository includes `prometheus-client` as a dependency. Instrumentation hooks should be added in
  `backend/app/main.py` or dedicated middleware before enabling scraping in production.
- Document metric names and scrape configuration here after implementation.

## Alerting Recommendations

1. **Incident Volume**: Alert when incident count exceeds threshold within a window (derive from
   `/api/telemetry/status` or database queries).
2. **Agent Health**: Monitor agent enrollment status and heartbeat failures via `/api/orchestrator/status`.
3. **ML Performance**: Alert on model accuracy degradation, concept drift detection, or training failures.
4. **Federated Learning**: Monitor participant connectivity and model aggregation status.
5. **Distributed Systems**: Alert on Kafka/Redis connectivity issues or node failures.
6. **Authentication Failures**: Monitor rate of 401/403 responses to detect credential issues or attacks.
7. **Scheduler Failures**: Alert on background job failures, queue backlogs, or execution timeouts.
8. **Resource Usage**: Monitor memory, CPU, and storage usage across distributed components.
9. **Tenant Isolation**: Alert on cross-tenant data access attempts or isolation failures.
10. **WebSocket/Webhook Issues**: Track connection failures and message delivery problems.

## Performance Monitoring

- **Response Times**: Monitor API endpoint latency, especially for incident processing and ML inference.
- **Throughput**: Track events processed per second, incidents created, and workflow executions.
- **Resource Utilization**: Monitor database connections, cache hit rates, and distributed queue depths.
- **Scalability Metrics**: Track auto-scaling events and load balancer distribution.

Update this document as monitoring integrations mature (Prometheus exporters, OpenTelemetry traces,
distributed tracing, log aggregation pipelines, etc.).
