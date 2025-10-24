"""
Network Discovery Engine

Comprehensive network scanning and asset discovery for automated
agent deployment and network mapping.

Features:
- ICMP host discovery
- TCP/UDP port scanning
- Service identification
- OS fingerprinting
- Network topology mapping
"""

import asyncio
import ipaddress
import socket
import subprocess
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
import concurrent.futures
import platform

logger = logging.getLogger(__name__)


class NetworkDiscoveryEngine:
    """
    Multi-phase network discovery engine.
    
    Phase 1: ICMP sweep for live host detection
    Phase 2: TCP/UDP port scanning for service identification
    Phase 3: Service fingerprinting (banners, versions)
    Phase 4: OS fingerprinting and classification
    Phase 5: Network dependency mapping
    """
    
    def __init__(self, timeout: int = 2, max_workers: int = 50):
        self.timeout = timeout
        self.max_workers = max_workers
        self.discovered_hosts: List[Dict] = []
        
    async def comprehensive_scan(
        self,
        network_ranges: List[str],
        port_ranges: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Perform comprehensive network scan.
        
        Args:
            network_ranges: List of network CIDR ranges (e.g., ["10.0.10.0/24"])
            port_ranges: Optional list of ports to scan (default: common ports)
            
        Returns:
            List of discovered hosts with details
        """
        logger.info(f"Starting comprehensive scan of {len(network_ranges)} network(s)")
        
        # Default common ports
        if port_ranges is None:
            port_ranges = [
                22, 80, 443, 445, 3389,  # Common services
                135, 139, 389, 636, 1433,  # Windows/AD
                3306, 5432, 6379, 8080, 8443  # Databases/Web
            ]
        
        all_hosts = []
        
        # Phase 1: ICMP sweep
        logger.info("Phase 1: ICMP host discovery")
        live_hosts = await self._icmp_sweep(network_ranges)
        logger.info(f"Found {len(live_hosts)} live hosts")
        
        # Phase 2: Port scanning
        logger.info("Phase 2: Port scanning")
        for host in live_hosts:
            host_details = await self._scan_host_ports(host, port_ranges)
            all_hosts.append(host_details)
        
        # Phase 3: Service fingerprinting
        logger.info("Phase 3: Service fingerprinting")
        for host in all_hosts:
            await self._fingerprint_services(host)
        
        # Phase 4: OS fingerprinting
        logger.info("Phase 4: OS fingerprinting")
        for host in all_hosts:
            await self._fingerprint_os(host)
        
        self.discovered_hosts = all_hosts
        logger.info(f"Scan complete: {len(all_hosts)} hosts discovered")
        
        return all_hosts
    
    async def _icmp_sweep(self, network_ranges: List[str]) -> List[str]:
        """
        ICMP sweep to find live hosts.
        
        Args:
            network_ranges: CIDR network ranges
            
        Returns:
            List of live host IP addresses
        """
        live_hosts = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for network_range in network_ranges:
                network = ipaddress.ip_network(network_range, strict=False)
                
                for ip in network.hosts():
                    futures.append(
                        executor.submit(self._ping_host, str(ip))
                    )
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    live_hosts.append(result)
        
        return live_hosts
    
    def _ping_host(self, ip: str) -> Optional[str]:
        """
        Ping a single host to check if it's alive.
        
        Args:
            ip: IP address to ping
            
        Returns:
            IP address if host is alive, None otherwise
        """
        try:
            # Determine ping command based on OS
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", "-w", str(self.timeout * 1000), ip]
            else:
                cmd = ["ping", "-c", "1", "-W", str(self.timeout), ip]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout + 1
            )
            
            if result.returncode == 0:
                logger.debug(f"Host alive: {ip}")
                return ip
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Host not responding: {ip} - {e}")
        
        return None
    
    async def _scan_host_ports(
        self,
        ip: str,
        ports: List[int]
    ) -> Dict:
        """
        Scan ports on a specific host.
        
        Args:
            ip: IP address to scan
            ports: List of port numbers
            
        Returns:
            Host details with open ports
        """
        host_details = {
            "ip": ip,
            "hostname": await self._resolve_hostname(ip),
            "ports": [],
            "services": {},
            "os_type": "unknown",
            "discovered_at": datetime.utcnow().isoformat()
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self._check_port, ip, port): port
                for port in ports
            }
            
            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                is_open = future.result()
                if is_open:
                    host_details["ports"].append(port)
        
        host_details["ports"].sort()
        return host_details
    
    def _check_port(self, ip: str, port: int) -> bool:
        """
        Check if a specific port is open.
        
        Args:
            ip: IP address
            port: Port number
            
        Returns:
            True if port is open, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                logger.debug(f"Port {port} open on {ip}")
                return True
        except Exception as e:
            logger.debug(f"Error checking port {port} on {ip}: {e}")
        
        return False
    
    async def _resolve_hostname(self, ip: str) -> Optional[str]:
        """
        Resolve hostname for an IP address.
        
        Args:
            ip: IP address
            
        Returns:
            Hostname if resolvable, None otherwise
        """
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except Exception:
            return None
    
    async def _fingerprint_services(self, host: Dict):
        """
        Identify services running on open ports.
        
        Args:
            host: Host details dictionary (modified in-place)
        """
        # Common service port mappings
        service_map = {
            22: "SSH",
            80: "HTTP",
            443: "HTTPS",
            445: "SMB",
            3389: "RDP",
            135: "MSRPC",
            139: "NetBIOS",
            389: "LDAP",
            636: "LDAPS",
            1433: "MSSQL",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP-Alt",
            8443: "HTTPS-Alt"
        }
        
        for port in host["ports"]:
            if port in service_map:
                service_name = service_map[port]
                host["services"][port] = {
                    "name": service_name,
                    "version": "unknown",
                    "banner": await self._grab_banner(host["ip"], port)
                }
    
    async def _grab_banner(self, ip: str, port: int) -> Optional[str]:
        """
        Attempt to grab service banner.
        
        Args:
            ip: IP address
            port: Port number
            
        Returns:
            Service banner if available
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=self.timeout
            )
            
            # Read banner (first 1024 bytes)
            banner = await asyncio.wait_for(
                reader.read(1024),
                timeout=self.timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            return banner.decode('utf-8', errors='ignore').strip()
        except Exception:
            return None
    
    async def _fingerprint_os(self, host: Dict):
        """
        Fingerprint operating system.
        
        Args:
            host: Host details dictionary (modified in-place)
        """
        # Simple heuristics based on open ports
        open_ports = set(host["ports"])
        
        # Windows indicators
        windows_ports = {135, 139, 445, 3389}
        if windows_ports & open_ports:
            host["os_type"] = "Windows"
            
            # Check for specific Windows roles
            if 389 in open_ports or 636 in open_ports:
                host["os_role"] = "Domain Controller"
            elif 3389 in open_ports:
                host["os_role"] = "Windows Workstation/Server"
        
        # Linux indicators
        elif 22 in open_ports:
            host["os_type"] = "Linux/Unix"
            
            # Check for server roles
            if 80 in open_ports or 443 in open_ports:
                host["os_role"] = "Web Server"
            elif 3306 in open_ports or 5432 in open_ports:
                host["os_role"] = "Database Server"
            else:
                host["os_role"] = "Linux Server"
        
        # Try TTL-based detection
        ttl = await self._get_ttl(host["ip"])
        if ttl:
            if ttl <= 64:
                host["os_type"] = "Linux/Unix" if host["os_type"] == "unknown" else host["os_type"]
            elif ttl <= 128:
                host["os_type"] = "Windows" if host["os_type"] == "unknown" else host["os_type"]
    
    async def _get_ttl(self, ip: str) -> Optional[int]:
        """
        Get TTL from ping response for OS fingerprinting.
        
        Args:
            ip: IP address
            
        Returns:
            TTL value if available
        """
        try:
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", ip]
            else:
                cmd = ["ping", "-c", "1", ip]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout + 1
            )
            
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                # Parse TTL from output
                for line in output.split('\n'):
                    if 'ttl=' in line.lower():
                        ttl_str = line.lower().split('ttl=')[1].split()[0]
                        return int(ttl_str)
        except Exception:
            pass
        
        return None
    
    async def generate_deployment_matrix(
        self,
        classified_assets: List[Dict]
    ) -> Dict:
        """
        Create comprehensive agent deployment plan.
        
        Args:
            classified_assets: List of classified hosts
            
        Returns:
            Deployment matrix with recommendations
        """
        deployment_plan = {
            "total_assets": len(classified_assets),
            "by_os_type": {},
            "by_role": {},
            "deployment_methods": {},
            "priority_groups": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        for asset in classified_assets:
            os_type = asset.get("os_type", "unknown")
            os_role = asset.get("os_role", "unknown")
            
            # Count by OS type
            deployment_plan["by_os_type"][os_type] = \
                deployment_plan["by_os_type"].get(os_type, 0) + 1
            
            # Count by role
            deployment_plan["by_role"][os_role] = \
                deployment_plan["by_role"].get(os_role, 0) + 1
            
            # Determine deployment method
            if os_type == "Windows":
                if os_role == "Domain Controller":
                    method = "PowerShell Remoting + GPO"
                    priority = "critical"
                else:
                    method = "Group Policy Deployment"
                    priority = "high"
            elif os_type == "Linux/Unix":
                method = "Ansible Playbook"
                priority = "high" if "Server" in os_role else "medium"
            else:
                method = "Manual Installation"
                priority = "low"
            
            deployment_plan["deployment_methods"][method] = \
                deployment_plan["deployment_methods"].get(method, 0) + 1
            
            deployment_plan["priority_groups"][priority].append({
                "ip": asset["ip"],
                "hostname": asset.get("hostname"),
                "os_type": os_type,
                "os_role": os_role,
                "deployment_method": method
            })
        
        return deployment_plan
    
    def get_summary_report(self) -> str:
        """
        Generate human-readable summary report.
        
        Returns:
            Formatted summary string
        """
        if not self.discovered_hosts:
            return "No hosts discovered yet. Run comprehensive_scan() first."
        
        report_lines = [
            "=" * 70,
            "NETWORK DISCOVERY SUMMARY",
            "=" * 70,
            "",
            f"Total Hosts: {len(self.discovered_hosts)}",
            "",
            "By Operating System:",
        ]
        
        # Count by OS
        os_counts = {}
        role_counts = {}
        
        for host in self.discovered_hosts:
            os_type = host.get("os_type", "unknown")
            os_role = host.get("os_role", "unknown")
            
            os_counts[os_type] = os_counts.get(os_type, 0) + 1
            role_counts[os_role] = role_counts.get(os_role, 0) + 1
        
        for os_type, count in sorted(os_counts.items()):
            report_lines.append(f"  • {os_type}: {count}")
        
        report_lines.extend([
            "",
            "By Role:",
        ])
        
        for role, count in sorted(role_counts.items()):
            report_lines.append(f"  • {role}: {count}")
        
        report_lines.extend([
            "",
            "Discovered Hosts:",
            ""
        ])
        
        for host in self.discovered_hosts:
            hostname = host.get("hostname") or "unknown"
            os_type = host.get("os_type", "unknown")
            os_role = host.get("os_role", "unknown")
            port_count = len(host.get("ports", []))
            
            report_lines.append(
                f"  • {host['ip']:15s} | {hostname:30s} | {os_type:10s} | "
                f"{os_role:20s} | {port_count} ports"
            )
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)


async def scan_azure_network(
    vnet_name: str,
    resource_group: str,
    subnet_name: str = "corp-network-subnet"
) -> List[Dict]:
    """
    Convenience function to scan Azure VNet subnet.
    
    Args:
        vnet_name: Azure VNet name
        resource_group: Azure resource group
        subnet_name: Subnet to scan
        
    Returns:
        List of discovered hosts
    """
    try:
        import subprocess
        
        # Get subnet address prefix using Azure CLI
        result = subprocess.run(
            [
                "az", "network", "vnet", "subnet", "show",
                "--resource-group", resource_group,
                "--vnet-name", vnet_name,
                "--name", subnet_name,
                "--query", "addressPrefix",
                "-o", "tsv"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            subnet_cidr = result.stdout.strip()
            
            logger.info(f"Scanning Azure subnet: {subnet_cidr}")
            
            scanner = NetworkDiscoveryEngine()
            hosts = await scanner.comprehensive_scan([subnet_cidr])
            
            return hosts
        else:
            logger.error(f"Failed to get subnet info: {result.stderr}")
            return []
    
    except Exception as e:
        logger.error(f"Error scanning Azure network: {e}")
        return []


if __name__ == "__main__":
    # Test the scanner
    async def test_scanner():
        scanner = NetworkDiscoveryEngine()
        
        # Scan a small network (adjust for your environment)
        hosts = await scanner.comprehensive_scan(["192.168.1.0/28"])
        
        print(scanner.get_summary_report())
        
        # Generate deployment matrix
        if hosts:
            matrix = await scanner.generate_deployment_matrix(hosts)
            print("\nDeployment Matrix:")
            print(f"  Critical Priority: {len(matrix['priority_groups']['critical'])}")
            print(f"  High Priority: {len(matrix['priority_groups']['high'])}")
            print(f"  Medium Priority: {len(matrix['priority_groups']['medium'])}")
            print(f"  Low Priority: {len(matrix['priority_groups']['low'])}")
    
    asyncio.run(test_scanner())

