# Multi-stage build for Mini-XDR Frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Build arguments for Next.js environment variables
ARG NEXT_PUBLIC_API_BASE=http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
ARG NEXT_PUBLIC_API_URL=http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
ARG NEXT_PUBLIC_API_KEY=demo-minixdr-api-key

# Set environment variables for build
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy application code
COPY . .

# Build Next.js application
RUN npm run build

# Production stage
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production

# Copy built application from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./next.config.js

# Expose port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Run application
CMD ["npm", "start"]



