# 🧪 Testing & Validation Scripts

Scripts for testing Mini-XDR functionality, adaptive detection, and system validation.

## Scripts Overview

### 🔬 Adaptive Detection Testing

#### `simple-test-adaptive.sh`
**Simple adaptive detection test**
- **Purpose**: Test adaptive detection system with clean JSON events
- **Usage**: `./simple-test-adaptive.sh`
- **Features**: Single web attacks, rapid sequences, incident verification

#### `simulate-advanced-attack-chain.sh`
**Advanced attack chain simulation**
- **Purpose**: Simulate complete multi-phase APT-style attacks
- **Usage**: `./simulate-advanced-attack-chain.sh`
- **Features**: Reconnaissance → Exploitation → Persistence → Exfiltration

### 🔍 System Validation

#### `verify_ip_blocks.py`
**IP block verification**
- **Purpose**: Verify IP blocks on T-Pot honeypot system
- **Usage**: `python3 verify_ip_blocks.py [optional_ip]`
- **Features**: iptables analysis, SSH connectivity, block status verification

## Usage Examples

### Basic Adaptive Detection Testing
```bash
# Test basic adaptive detection
./testing/simple-test-adaptive.sh

# Expected output:
# - Events processed
# - Incidents detected
# - Learning pipeline triggered
```

### Advanced Attack Simulation
```bash
# Simulate complete attack chain
./testing/simulate-advanced-attack-chain.sh

# This creates a realistic multi-phase attack:
# 1. Reconnaissance phase
# 2. SQL injection testing
# 3. Database access
# 4. Privilege escalation
# 5. Data exfiltration
# 6. Persistence establishment
# 7. Lateral movement preparation
```

### IP Block Verification
```bash
# Check all blocked IPs
python3 testing/verify_ip_blocks.py

# Check specific IP
python3 testing/verify_ip_blocks.py 192.168.1.100

# Verify containment actions worked
python3 testing/verify_ip_blocks.py ATTACKER_IP
```

## Test Categories

### 🎯 Detection Testing
- **Signature-based**: Known attack patterns
- **Behavioral**: Adaptive pattern recognition  
- **ML-based**: Machine learning detection
- **Zero-day**: Unknown attack methods

### 📊 Response Testing
- **Incident Creation**: Verify incidents are created
- **Risk Scoring**: Confirm appropriate risk levels
- **Containment**: Test IP blocking and isolation
- **AI Analysis**: Validate automated analysis

### 🔄 System Integration Testing
- **Multi-source Ingestion**: Various log sources
- **API Authentication**: HMAC signature validation
- **Real-time Processing**: Live event handling
- **Dashboard Updates**: UI synchronization

## Validation Workflows

### Development Testing
```bash
# 1. Test basic functionality
./testing/simple-test-adaptive.sh

# 2. Verify detection works
curl http://localhost:8000/incidents

# 3. Test containment
python3 testing/verify_ip_blocks.py
```

### Pre-deployment Validation
```bash
# 1. Complete attack chain test
./testing/simulate-advanced-attack-chain.sh

# 2. Verify all phases detected
curl http://localhost:8000/incidents | jq '.[0]'

# 3. Test SOC response actions
# (Use dashboard to test Block IP, Threat Intel, etc.)
```

### Production Readiness
```bash
# 1. Adaptive detection validation
./testing/simple-test-adaptive.sh

# 2. Advanced attack detection
./testing/simulate-advanced-attack-chain.sh

# 3. System integration check
python3 testing/verify_ip_blocks.py

# 4. Dashboard functionality test
# Visit http://localhost:3000 and test all features
```

## Expected Outcomes

### Successful Tests Show
- ✅ Incidents created with correct source IPs
- ✅ Attack types properly categorized
- ✅ Risk scores calculated appropriately
- ✅ Behavioral patterns detected
- ✅ Containment actions functional
- ✅ AI analysis triggered

### Test Metrics
- **Detection Rate**: % of attacks detected
- **False Positive Rate**: % of benign traffic flagged
- **Response Time**: Time from attack to detection
- **Containment Success**: % of successful IP blocks

## Integration with Mini-XDR

### Testing Pipeline
```
Generate Test Data → Send to API → Verify Detection → Test Response → Validate Results
```

### Dependencies
- **Backend Running**: Required for API testing
- **Authentication**: Uses auth/send_signed_request.py
- **T-Pot Access**: For IP block verification
- **Database**: For incident persistence

---

**Status**: Production Ready  
**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR Testing Team
