"""
Asset Classification Module

ML-based device classification using network characteristics,
port patterns, and service fingerprinting.
"""

import logging
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class AssetClassifier:
    """
    Classifies discovered network assets into categories.
    
    Categories:
    - Domain Controller
    - Windows Workstation  
    - Windows Server
    - Linux Server
    - Database Server
    - Web Server
    - Network Device
    - Unknown
    """
    
    def __init__(self):
        # Port signature patterns for classification
        self.signatures = {
            "Domain Controller": {
                "required_ports": [389],  # LDAP
                "common_ports": [88, 135, 139, 445, 636, 3268, 3269],
                "os_type": "Windows"
            },
            "Windows Server": {
                "required_ports": [445],  # SMB
                "common_ports": [135, 139, 3389],
                "os_type": "Windows"
            },
            "Windows Workstation": {
                "required_ports": [445],  # SMB
                "common_ports": [135, 139, 3389],
                "os_type": "Windows",
                "exclude_ports": [389, 636]  # Not a DC
            },
            "Database Server": {
                "any_ports": [1433, 3306, 5432, 1521, 27017],  # MSSQL, MySQL, Postgres, Oracle, MongoDB
                "os_type": "any"
            },
            "Web Server": {
                "any_ports": [80, 443, 8080, 8443],
                "os_type": "any"
            },
            "Linux Server": {
                "required_ports": [22],  # SSH
                "os_type": "Linux/Unix"
            },
            "Network Device": {
                "any_ports": [161, 162, 23],  # SNMP, Telnet
                "os_type": "any"
            }
        }
    
    def classify(self, host: Dict) -> str:
        """
        Classify a discovered host.
        
        Args:
            host: Host details from network scanner
            
        Returns:
            Classification string
        """
        open_ports = set(host.get("ports", []))
        os_type = host.get("os_type", "unknown")
        
        # Check each signature
        scores = {}
        
        for classification, signature in self.signatures.items():
            score = 0
            
            # Check OS type match
            sig_os = signature.get("os_type", "any")
            if sig_os != "any" and os_type != "unknown":
                if sig_os != os_type:
                    continue  # OS type mismatch, skip
                else:
                    score += 10  # OS type match bonus
            
            # Check required ports
            required = set(signature.get("required_ports", []))
            if required and required.issubset(open_ports):
                score += 50
            elif required and not required.issubset(open_ports):
                continue  # Missing required ports, skip
            
            # Check if any required port is present
            any_ports = set(signature.get("any_ports", []))
            if any_ports and any_ports & open_ports:
                score += 30
            
            # Check common ports
            common = set(signature.get("common_ports", []))
            common_matches = common & open_ports
            score += len(common_matches) * 5
            
            # Check exclude ports
            exclude = set(signature.get("exclude_ports", []))
            if exclude and exclude & open_ports:
                score -= 100  # Penalty for excluded ports
            
            if score > 0:
                scores[classification] = score
        
        # Return highest scoring classification
        if scores:
            classification = max(scores, key=scores.get)
            confidence = min(scores[classification] / 100.0, 1.0)
            
            logger.info(
                f"Classified {host['ip']} as {classification} "
                f"(confidence: {confidence:.2f})"
            )
            
            # Add classification to host object
            host["classification"] = classification
            host["classification_confidence"] = confidence
            
            return classification
        
        return "Unknown"
    
    def classify_and_profile(self, discovered_hosts: List[Dict]) -> List[Dict]:
        """
        Classify all discovered hosts and generate profiles.
        
        Args:
            discovered_hosts: List of hosts from network scanner
            
        Returns:
            List of classified hosts with profiles
        """
        classified_hosts = []
        
        for host in discovered_hosts:
            classification = self.classify(host)
            
            # Generate deployment profile
            host["deployment_profile"] = self._generate_deployment_profile(
                host,
                classification
            )
            
            classified_hosts.append(host)
        
        return classified_hosts
    
    def _generate_deployment_profile(
        self,
        host: Dict,
        classification: str
    ) -> Dict:
        """
        Generate agent deployment profile for a host.
        
        Args:
            host: Host details
            classification: Asset classification
            
        Returns:
            Deployment profile with method, credentials, etc.
        """
        profile = {
            "agent_compatible": True,
            "deployment_method": "Unknown",
            "requirements": [],
            "risks": [],
            "estimated_time": "5 minutes"
        }
        
        if classification == "Domain Controller":
            profile["deployment_method"] = "PowerShell Remoting"
            profile["requirements"] = [
                "Administrator credentials",
                "WinRM enabled",
                "PowerShell 5.1+",
                "Network connectivity to DC"
            ]
            profile["risks"] = [
                "Critical system - test in maintenance window",
                "Reboot may be required",
                "Impact to domain authentication if issues occur"
            ]
            profile["estimated_time"] = "10 minutes"
            profile["priority"] = "critical"
        
        elif classification in ["Windows Server", "Windows Workstation"]:
            profile["deployment_method"] = "Group Policy Deployment"
            profile["requirements"] = [
                "Domain-joined system",
                "GPO deployment configured",
                "Network connectivity"
            ]
            profile["risks"] = ["Minimal - standard deployment"]
            profile["estimated_time"] = "5 minutes"
            profile["priority"] = "high"
        
        elif classification == "Linux Server":
            profile["deployment_method"] = "Ansible Playbook"
            profile["requirements"] = [
                "SSH access",
                "Python 3.6+",
                "sudo privileges"
            ]
            profile["risks"] = ["Minimal - non-invasive installation"]
            profile["estimated_time"] = "3 minutes"
            profile["priority"] = "high"
        
        elif classification == "Database Server":
            profile["deployment_method"] = "Manual Installation"
            profile["requirements"] = [
                "Administrative access",
                "Database maintenance window",
                "Backup completed"
            ]
            profile["risks"] = [
                "Critical system - careful testing required",
                "Potential performance impact",
                "Test in non-production first"
            ]
            profile["estimated_time"] = "15 minutes"
            profile["priority"] = "high"
        
        elif classification == "Web Server":
            profile["deployment_method"] = "Container Sidecar or Service Install"
            profile["requirements"] = [
                "Root/admin access",
                "Network connectivity",
                "Firewall rules configured"
            ]
            profile["risks"] = ["Minimal - standard deployment"]
            profile["estimated_time"] = "5 minutes"
            profile["priority"] = "medium"
        
        else:
            profile["agent_compatible"] = False
            profile["deployment_method"] = "Not Supported"
            profile["priority"] = "low"
        
        return profile
    
    def get_deployment_summary(self, classified_hosts: List[Dict]) -> Dict:
        """
        Generate deployment summary statistics.
        
        Args:
            classified_hosts: List of classified hosts
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_hosts": len(classified_hosts),
            "compatible_hosts": 0,
            "by_priority": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "by_method": {},
            "estimated_total_time": 0
        }
        
        for host in classified_hosts:
            profile = host.get("deployment_profile", {})
            
            if profile.get("agent_compatible"):
                summary["compatible_hosts"] += 1
            
            priority = profile.get("priority", "low")
            summary["by_priority"][priority] += 1
            
            method = profile.get("deployment_method", "Unknown")
            summary["by_method"][method] = summary["by_method"].get(method, 0) + 1
            
            # Parse estimated time (assume format: "X minutes")
            time_str = profile.get("estimated_time", "0 minutes")
            minutes = int(time_str.split()[0]) if time_str.split()[0].isdigit() else 5
            summary["estimated_total_time"] += minutes
        
        return summary


if __name__ == "__main__":
    # Test classification
    test_hosts = [
        {
            "ip": "10.0.10.10",
            "hostname": "dc01.minicorp.local",
            "ports": [53, 88, 135, 139, 389, 445, 636, 3268, 3269, 3389],
            "os_type": "Windows"
        },
        {
            "ip": "10.0.10.20",
            "hostname": "ws01.minicorp.local",
            "ports": [135, 139, 445, 3389],
            "os_type": "Windows"
        },
        {
            "ip": "10.0.10.30",
            "hostname": "fileserver01",
            "ports": [22, 139, 445],
            "os_type": "Linux/Unix"
        },
        {
            "ip": "10.0.10.40",
            "hostname": "db01",
            "ports": [22, 3306],
            "os_type": "Linux/Unix"
        }
    ]
    
    classifier = AssetClassifier()
    
    print("Testing Asset Classifier")
    print("=" * 70)
    print()
    
    classified = classifier.classify_and_profile(test_hosts)
    
    for host in classified:
        print(f"Host: {host['ip']} ({host.get('hostname', 'unknown')})")
        print(f"  Classification: {host.get('classification', 'Unknown')}")
        print(f"  Confidence: {host.get('classification_confidence', 0):.2f}")
        print(f"  Deployment Method: {host['deployment_profile']['deployment_method']}")
        print(f"  Priority: {host['deployment_profile'].get('priority', 'unknown')}")
        print()
    
    summary = classifier.get_deployment_summary(classified)
    print("Deployment Summary:")
    print(f"  Total Hosts: {summary['total_hosts']}")
    print(f"  Compatible: {summary['compatible_hosts']}")
    print(f"  Estimated Time: {summary['estimated_total_time']} minutes")
    print(f"  By Priority: {summary['by_priority']}")

