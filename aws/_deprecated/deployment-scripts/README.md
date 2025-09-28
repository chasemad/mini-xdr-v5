# 📦 Deprecated Deployment Scripts

**Status**: ⚠️ DEPRECATED - Use `../../start-mini-xdr-aws-v2.sh` instead

## Scripts in this folder:

### `deploy-automated-production.sh`
- **Purpose**: Automated production environment setup
- **Deprecated**: September 2025
- **Replacement**: `start-mini-xdr-aws-v2.sh testing` or `start-mini-xdr-aws-v2.sh live`

### `deploy-complete-aws-ml-system.sh`
- **Purpose**: Complete ML system with SageMaker deployment
- **Deprecated**: September 2025  
- **Replacement**: `start-mini-xdr-aws-v2.sh` + individual ML training scripts

### `deploy-mini-xdr-code.sh`
- **Purpose**: Application code deployment to EC2
- **Deprecated**: September 2025
- **Replacement**: Code deployment handled by `start-mini-xdr-aws-v2.sh`

### `deploy-secure-mini-xdr.sh`
- **Purpose**: Secure infrastructure deployment
- **Deprecated**: September 2025
- **Replacement**: Security built into `start-mini-xdr-aws-v2.sh`

### `deploy-secure-ml-production.sh`
- **Purpose**: Secure ML production environment
- **Deprecated**: September 2025
- **Replacement**: `start-mini-xdr-aws-v2.sh deploy`

## Migration Path
```bash
# Old approach:
./deploy-complete-aws-ml-system.sh

# New approach:
./start-mini-xdr-aws-v2.sh testing  # Start system
./start-mini-xdr-aws-v2.sh deploy   # Deploy trained models
```
