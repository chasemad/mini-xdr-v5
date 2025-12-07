# ============================================
# Mini-XDR Backend - Production Dockerfile
# ============================================
# Multi-stage build for security and size optimization
# Includes fixed auth.py with direct bcrypt implementation

# ============ Builder Stage ============
FROM python:3.11.9-slim AS builder

LABEL stage=builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies to user directory
RUN pip install --user --no-cache-dir --no-warn-script-location \
    -r requirements.txt

# ============ Production Stage ============
FROM python:3.11.9-slim

# Metadata labels following OCI Image Spec
LABEL org.opencontainers.image.title="Mini-XDR Backend" \
      org.opencontainers.image.description="Enterprise AI-Powered XDR Platform Backend" \
      org.opencontainers.image.vendor="Mini-XDR" \
      org.opencontainers.image.authors="Mini-XDR Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/chasemad/mini-xdr-v5" \
      org.opencontainers.image.documentation="https://github.com/chasemad/mini-xdr-v5/blob/main/README.md"

# Security: Create non-root user
RUN groupadd -r -g 1000 xdr && \
    useradd -r -u 1000 -g xdr -m -d /app -s /sbin/nologin \
    -c "Mini-XDR Application User" xdr

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder --chown=xdr:xdr /root/.local /home/xdr/.local

# Make sure scripts in .local are usable
ENV PATH=/home/xdr/.local/bin:$PATH \
    PYTHONUSERBASE=/home/xdr/.local

# Copy application code (includes fixed auth.py)
COPY --chown=xdr:xdr . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/honeypot_configs /app/models /tmp/matplotlib && \
    chown -R xdr:xdr /app/data /app/models /tmp/matplotlib

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HONEYPOT_CONFIG_DIR=/app/data/honeypot_configs \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/app \
    TMPDIR=/tmp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER xdr:xdr

# Run application with uvicorn
# Note: Single worker in container, scale with Kubernetes replicas
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
