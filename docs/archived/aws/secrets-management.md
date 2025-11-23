# Secrets Management

Mini-XDR expects sensitive values (API keys, agent credentials, honeypot SSH keys) to be supplied via
environment variables. The backend can optionally bootstrap from AWS Secrets Manager.

## Local Development

1. Duplicate `backend/env.example` to `backend/.env` and replace placeholder entries. Keep secrets
   scoped to your local machine.
2. Generate secure random keys for required secrets:
   ```bash
   # Generate API key (64+ characters)
   openssl rand -hex 32

   # Generate JWT secret (32+ characters)
   openssl rand -hex 32

   # Generate encryption key (32+ characters)
   openssl rand -hex 32
   ```
3. Add the backend API key to `frontend/.env.local` as `NEXT_PUBLIC_API_KEY` if you want the UI to
   call secured endpoints without manual headers.
4. Store SSH private keys outside the repository; reference them using absolute paths in
   `HONEYPOT_SSH_KEY`.

## AWS Secrets Manager Integration

`backend/app/config.py` calls `_load_secrets_on_init()` before `Settings` instantiation when the
environment variable `SECRETS_MANAGER_ENABLED=true`. The helper `backend/app/secrets_manager.py`
expects:

- `AWS_REGION`
- `SECRETS_MANAGER_SECRET_ID` or `SECRETS_MANAGER_PREFIX`

Secrets are returned as a flat dictionary and pushed into `os.environ` before Pydantic settings load.
Any entries defined both locally and in Secrets Manager use the manager value.

### Recommended Secret Keys

| Secret | Purpose |
| --- | --- |
| `API_KEY` | API key for UI/automation access. |
| `JWT_SECRET_KEY` | Signing key for JWT authentication (`backend/app/auth.py`). |
| `ENCRYPTION_KEY` | Key for encrypting sensitive data in database. |
| `OPENAI_API_KEY`, `XAI_API_KEY` | Required for AI advisor and NLP workflow enhancements. |
| `ABUSEIPDB_API_KEY`, `VIRUSTOTAL_API_KEY` | Threat intel enrichment. |
| `AGENT_API_KEY` | API key for agent operations. |
| `{AGENT_NAME}_AGENT_*` | HMAC credentials for each agent (device_id, public_id, hmac_key, secret). |
| `DATABASE_URL` | Managed Postgres connection string for production. |
| `HONEYPOT_*`, `TPOT_*` | T-Pot honeypot configuration and credentials. |
| `SECRETS_MANAGER_SECRET_ID` | AWS Secrets Manager secret identifier. |

## Rotation & Auditing

- Rotate API keys and agent secrets monthly; update both the secret store and deployed environments.
- Leverage IAM policies to limit access to the secrets prefix. The backend only requires `secretsmanager:GetSecretValue`.
- Keep an audit log of rotations in `change-control/audit-log.md` and update applications promptly.

## Local Testing of AWS Integration

Use AWS CLI to put a JSON payload into Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name mini-xdr/dev/config \
  --secret-string '{"API_KEY":"dev-key","JWT_SECRET_KEY":"local-jwt","ENCRYPTION_KEY":"dev-encrypt-key"}'
```

Then run the backend with:

```bash
export SECRETS_MANAGER_ENABLED=true
export SECRETS_MANAGER_SECRET_ID=mini-xdr/dev/config
uvicorn backend.app.main:app --reload
```

Verify that `settings.api_key` resolves to the secret store value.
