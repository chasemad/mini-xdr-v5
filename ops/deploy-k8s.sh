#!/bin/bash

# Enhanced Mini-XDR Kubernetes Deployment Script
set -e

NAMESPACE="mini-xdr"
REGISTRY="${REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"

echo "🚀 Starting Enhanced Mini-XDR Kubernetes deployment..."

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl is not installed or not in PATH"
        exit 1
    fi
}

# Function to check if cluster is accessible
check_cluster() {
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    echo "✅ Connected to Kubernetes cluster"
}

# Function to build Docker images
build_images() {
    echo "🔨 Building Docker images..."
    
    # Build backend image
    echo "Building backend image..."
    docker build -f ops/Dockerfile.backend -t ${REGISTRY}/mini-xdr-backend:${VERSION} .
    
    # Build frontend image
    echo "Building frontend image..."
    docker build -f ops/Dockerfile.frontend -t ${REGISTRY}/mini-xdr-frontend:${VERSION} .
    
    # Build ingestion agent image
    echo "Building ingestion agent image..."
    docker build -f ops/Dockerfile.ingestion-agent -t ${REGISTRY}/mini-xdr-ingestion-agent:${VERSION} .
    
    echo "✅ Docker images built successfully"
}

# Function to push images to registry
push_images() {
    echo "📤 Pushing images to registry..."
    
    docker push ${REGISTRY}/mini-xdr-backend:${VERSION}
    docker push ${REGISTRY}/mini-xdr-frontend:${VERSION}
    docker push ${REGISTRY}/mini-xdr-ingestion-agent:${VERSION}
    
    echo "✅ Images pushed successfully"
}

# Function to create namespace
create_namespace() {
    echo "📁 Creating namespace..."
    kubectl apply -f ops/k8s/namespace.yaml
    echo "✅ Namespace created"
}

# Function to create secrets
create_secrets() {
    echo "🔐 Creating secrets..."
    
    # Check if secrets already exist
    if kubectl get secret mini-xdr-secrets -n ${NAMESPACE} &> /dev/null; then
        echo "⚠️  Secrets already exist, skipping creation"
        return
    fi
    
    # Prompt for required secrets
    echo "Please provide the following secrets:"
    
    read -s -p "OpenAI API Key (optional, press enter to skip): " OPENAI_API_KEY
    echo
    
    read -s -p "xAI API Key (optional, press enter to skip): " XAI_API_KEY
    echo
    
    read -s -p "Agent API Key (required): " AGENT_API_KEY
    echo
    
    if [ -z "$AGENT_API_KEY" ]; then
        echo "❌ Agent API Key is required"
        exit 1
    fi
    
    # Read SSH key file
    SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/xdrops_id_ed25519}"
    if [ ! -f "$SSH_KEY_PATH" ]; then
        echo "❌ SSH key not found at $SSH_KEY_PATH"
        echo "Please set SSH_KEY_PATH environment variable or ensure key exists"
        exit 1
    fi
    
    # Create secret
    kubectl create secret generic mini-xdr-secrets \
        --namespace=${NAMESPACE} \
        --from-literal=openai-api-key="${OPENAI_API_KEY}" \
        --from-literal=xai-api-key="${XAI_API_KEY}" \
        --from-literal=agent-api-key="${AGENT_API_KEY}" \
        --from-file=honeypot-ssh-key="${SSH_KEY_PATH}"
    
    echo "✅ Secrets created"
}

# Function to deploy resources
deploy_resources() {
    echo "🚀 Deploying Kubernetes resources..."
    
    # Apply in order
    kubectl apply -f ops/k8s/configmap.yaml
    kubectl apply -f ops/k8s/persistent-volumes.yaml
    kubectl apply -f ops/k8s/backend-deployment.yaml
    kubectl apply -f ops/k8s/frontend-deployment.yaml
    kubectl apply -f ops/k8s/ingestion-agent-daemonset.yaml
    
    # Apply ingress if specified
    if [ "$DEPLOY_INGRESS" = "true" ]; then
        echo "🌐 Deploying ingress..."
        kubectl apply -f ops/k8s/ingress.yaml
    fi
    
    echo "✅ Resources deployed"
}

# Function to wait for rollout
wait_for_rollout() {
    echo "⏳ Waiting for deployments to be ready..."
    
    kubectl rollout status deployment/mini-xdr-backend -n ${NAMESPACE} --timeout=300s
    kubectl rollout status deployment/mini-xdr-frontend -n ${NAMESPACE} --timeout=300s
    kubectl rollout status daemonset/mini-xdr-ingestion-agent -n ${NAMESPACE} --timeout=300s
    
    echo "✅ All deployments are ready"
}

# Function to show status
show_status() {
    echo "📊 Deployment status:"
    echo
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE}
    echo
    echo "Services:"
    kubectl get services -n ${NAMESPACE}
    echo
    echo "Ingress:"
    kubectl get ingress -n ${NAMESPACE} 2>/dev/null || echo "No ingress configured"
    echo
    
    # Get service URLs
    echo "🌐 Access URLs:"
    if [ "$DEPLOY_INGRESS" = "true" ]; then
        INGRESS_HOST=$(kubectl get ingress mini-xdr-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")
        if [ -n "$INGRESS_HOST" ]; then
            echo "Frontend: https://${INGRESS_HOST}"
            echo "Backend API: https://${INGRESS_HOST}/api"
        fi
    else
        echo "To access locally, run:"
        echo "  kubectl port-forward -n ${NAMESPACE} svc/mini-xdr-frontend-service 3000:3000"
        echo "  kubectl port-forward -n ${NAMESPACE} svc/mini-xdr-backend-service 8000:8000"
        echo "Then visit: http://localhost:3000"
    fi
}

# Main deployment function
main() {
    check_kubectl
    check_cluster
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --push)
                PUSH_IMAGES=true
                shift
                ;;
            --ingress)
                DEPLOY_INGRESS=true
                shift
                ;;
            --skip-secrets)
                SKIP_SECRETS=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --build         Build Docker images"
                echo "  --push          Push images to registry"
                echo "  --ingress       Deploy ingress resources"
                echo "  --skip-secrets  Skip secret creation"
                echo "  --registry REG  Docker registry (default: localhost:5000)"
                echo "  --version VER   Image version (default: latest)"
                echo "  -h, --help      Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    if [ "$BUILD_IMAGES" = "true" ]; then
        build_images
    fi
    
    if [ "$PUSH_IMAGES" = "true" ]; then
        push_images
    fi
    
    create_namespace
    
    if [ "$SKIP_SECRETS" != "true" ]; then
        create_secrets
    fi
    
    deploy_resources
    wait_for_rollout
    show_status
    
    echo
    echo "🎉 Enhanced Mini-XDR deployment completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Configure your honeypots to send logs to the ingestion agents"
    echo "2. Set up threat intelligence API keys in the secrets"
    echo "3. Train ML models with initial data"
    echo "4. Configure containment policies"
}

# Run main function
main "$@"
