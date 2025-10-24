# ============================================
# Mini-XDR Frontend - Production Dockerfile
# ============================================
# Multi-stage build with TypeScript support and security hardening

# ============ Builder Stage ============
FROM node:18-alpine AS builder

LABEL stage=builder

# Set working directory
WORKDIR /app

# Build arguments for Next.js public environment variables
# These get baked into the build
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_API_KEY=demo-minixdr-api-key

# Set environment variables for build
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE \
    NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY \
    NEXT_TELEMETRY_DISABLED=1

# Install dependencies based on lockfile
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy application source (includes TypeScript config)
COPY . .

# Build Next.js application
# This handles TypeScript compilation automatically
RUN npm run build

# ============ Production Stage ============
FROM node:18-alpine

# Metadata labels following OCI Image Spec
LABEL org.opencontainers.image.title="Mini-XDR Frontend" \
      org.opencontainers.image.description="Enterprise XDR Platform Web Interface" \
      org.opencontainers.image.vendor="Mini-XDR" \
      org.opencontainers.image.authors="chasemadrian@protonmail.com" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/chasemad/mini-xdr" \
      org.opencontainers.image.documentation="https://github.com/chasemad/mini-xdr/blob/main/README.md"

# Security: Create non-root user with home directory
# Handle case where group 1000 might already exist
RUN addgroup -g 1000 -S xdr 2>/dev/null || true && \
    adduser -u 1000 -S xdr -h /home/xdr 2>/dev/null || true && \
    mkdir -p /home/xdr/.npm /home/xdr/.cache && \
    chown -R xdr:xdr /home/xdr 2>/dev/null || chown -R 1000:1000 /home/xdr

# Set working directory
WORKDIR /app

# Copy package files
COPY --chown=xdr:xdr package.json package-lock.json ./

# Install production dependencies + TypeScript (needed for next.config.ts)
RUN npm ci --only=production --ignore-scripts && \
    npm install --save-exact typescript@5.9.3 && \
    npm cache clean --force

# Copy built application from builder (standalone mode)
# Next.js standalone mode creates a minimal server.js
COPY --from=builder --chown=xdr:xdr /app/.next ./.next
COPY --from=builder --chown=xdr:xdr /app/public ./public
# Copy next.config.ts for runtime (TypeScript supported in production)
COPY --from=builder --chown=xdr:xdr /app/next.config.ts ./next.config.ts

# Set environment variables
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME="0.0.0.0" \
    NEXT_TELEMETRY_DISABLED=1 \
    HOME=/home/xdr

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Switch to non-root user
USER xdr:xdr

# Run application
CMD ["npm", "start"]
