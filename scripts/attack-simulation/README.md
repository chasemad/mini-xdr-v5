# 🚨 Attack Simulation Scripts

This directory contains scripts for testing your Mini-XDR system's detection and response capabilities through controlled attack simulations.

## Scripts Overview

### 🎯 Main Attack Simulators

#### `attack_simulation.py`
**Comprehensive multi-vector attack simulator**
- **Purpose**: Full-featured attack simulation with multiple attack types
- **Features**: SQL injection, XSS, brute force, port scanning, reconnaissance
- **Usage**: `python3 attack_simulation.py --target <IP> [options]`
- **Best for**: Complete system testing and SOC training

#### `simple_attack_test.py`
**Focused attack test for quick validation**
- **Purpose**: Quick validation of Mini-XDR detection capabilities
- **Features**: Web attacks, brute force, directory traversal, reconnaissance
- **Usage**: `python3 simple_attack_test.py <TARGET_IP>`
- **Best for**: Quick functionality tests and development validation

### 🌐 Multi-IP Attack Scripts

#### `multi_ip_attack.sh`
**Advanced multi-source attack simulation**
- **Purpose**: Simulate attacks from multiple different IP addresses
- **Features**: Creates separate incidents, uses known malicious IPs
- **Usage**: Edit TARGET_IP in script, then `./multi_ip_attack.sh`
- **Best for**: Testing incident correlation and threat intelligence

#### `simple_multi_ip_attack.sh`
**Quick multi-IP attack test**
- **Purpose**: Simple multi-source attack simulation
- **Features**: Fast attacks from 5 different fake IP addresses
- **Usage**: Edit TARGET_IP in script, then `./simple_multi_ip_attack.sh`
- **Best for**: Quick multi-incident testing

### ⚡ Quick Test Scripts

#### `quick_attack.sh`
**Rapid attack test script**
- **Purpose**: Fast attack sequence for immediate testing
- **Features**: SQL injection, brute force, directory traversal, reconnaissance
- **Usage**: `./quick_attack.sh <TARGET_IP>`
- **Best for**: Development testing and quick validation

## Usage Examples

### Basic Testing
```bash
# Quick functionality test
./quick_attack.sh 192.168.1.100

# Simple focused test
python3 simple_attack_test.py 192.168.1.100

# Comprehensive test with options
python3 attack_simulation.py --target 192.168.1.100 --intensity medium --duration 300
```

### Advanced Testing
```bash
# Multi-IP incident creation
./simple_multi_ip_attack.sh  # Edit TARGET_IP first

# Full multi-vector simulation
python3 attack_simulation.py --target honeypot.example.com --intensity high --duration 600
```

### T-Pot Honeypot Testing
```bash
# Test T-Pot honeypot (after enabling Kali access)
python3 attack_simulation.py --target 34.193.101.171 --port 22 --intensity low
python3 simple_attack_test.py 34.193.101.171
```

## Expected Results

### Mini-XDR Detection
- **Incidents Created**: Each script should create 1+ incidents
- **Risk Scores**: High risk scores due to malicious patterns
- **Threat Intel Hits**: Known malicious user agents detected
- **ML Detection**: Attack patterns identified by ML models

### SOC Dashboard Features to Test
- **Block IP**: Test IP blocking functionality
- **Threat Intel**: Lookup IOCs and malicious indicators
- **Hunt Similar**: Find related attack patterns
- **Isolate Host**: Test containment actions
- **AI Analysis**: Review automated threat analysis

## Security Considerations

### ⚠️ IMPORTANT WARNINGS
- **Only use against systems you own or have explicit permission to test**
- **Never run against production systems without authorization**
- **These scripts generate real attack traffic that will be logged**
- **Some payloads may trigger security tools and alerts**

### Ethical Usage
- Use only in authorized testing environments
- Inform security teams before running tests
- Document all testing activities
- Follow responsible disclosure practices

## Customization

### Modifying Attack Patterns
Edit the payload arrays in each script:
- `sql_payloads[]` - SQL injection patterns
- `xss_payloads[]` - Cross-site scripting patterns
- `login_attempts[]` - Brute force credentials
- `web_paths[]` - Reconnaissance targets

### Adding New Attack Types
1. Add new payload arrays
2. Create attack functions
3. Integrate into main attack loops
4. Update documentation

### Adjusting Intensity
- **Low**: 2-5 second delays between attacks
- **Medium**: 0.5-2 second delays (default)
- **High**: 0.1-0.5 second delays (intensive)

## Integration with Mini-XDR

### Testing Workflow
1. **Start Mini-XDR**: Ensure backend and frontend are running
2. **Run Attack Script**: Execute chosen simulation script
3. **Monitor Dashboard**: Watch for incident creation in real-time
4. **Test Responses**: Use SOC actions to respond to incidents
5. **Analyze Results**: Review detection accuracy and response times

### Validation Checklist
- [ ] Incidents created with correct source IPs
- [ ] Attack types properly categorized
- [ ] Risk scores calculated appropriately
- [ ] Threat intelligence lookups working
- [ ] ML detection triggering correctly
- [ ] SOC response actions functional
- [ ] Logs captured and stored properly

---

**Last Updated**: September 16, 2025  
**Maintained by**: Mini-XDR Security Team  
**Status**: Production Ready


