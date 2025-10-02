"""
Intelligent System Detection for Adaptive Response Actions
Automatically detects target system capabilities and adapts response commands
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SystemType(Enum):
    """Detected system types"""
    UBUNTU = "ubuntu"
    DEBIAN = "debian"
    RHEL = "rhel"
    CENTOS = "centos"
    ALPINE = "alpine"
    TPOT = "tpot"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class FirewallType(Enum):
    """Detected firewall types"""
    UFW = "ufw"
    IPTABLES = "iptables"
    FIREWALLD = "firewalld"
    WINDOWS_FIREWALL = "windows_firewall"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class SystemCapabilities:
    """System capabilities and available tools"""
    system_type: SystemType
    firewall_type: FirewallType
    has_sudo: bool
    has_docker: bool
    has_systemctl: bool
    package_manager: str
    shell_type: str
    architecture: str
    available_tools: List[str]
    security_features: List[str]


class IntelligentSystemDetector:
    """
    Detects target system capabilities and adapts response commands accordingly
    """
    
    def __init__(self, responder_agent):
        self.responder = responder_agent
        self.logger = logging.getLogger(__name__)
        self._capabilities_cache = {}
        
    async def detect_system_capabilities(self, cache_key: str = "default") -> SystemCapabilities:
        """
        Comprehensively detect target system capabilities
        """
        if cache_key in self._capabilities_cache:
            self.logger.debug(f"Using cached capabilities for {cache_key}")
            return self._capabilities_cache[cache_key]
        
        self.logger.info("Detecting target system capabilities...")
        
        # Initialize default capabilities
        capabilities = SystemCapabilities(
            system_type=SystemType.UNKNOWN,
            firewall_type=FirewallType.UNKNOWN,
            has_sudo=False,
            has_docker=False,
            has_systemctl=False,
            package_manager="unknown",
            shell_type="unknown",
            architecture="unknown",
            available_tools=[],
            security_features=[]
        )
        
        # Run detection commands in parallel
        detection_tasks = [
            self._detect_os_type(),
            self._detect_firewall_type(),
            self._detect_package_manager(),
            self._detect_security_tools(),
            self._detect_system_info(),
            self._detect_container_environment()
        ]
        
        try:
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    if i == 0:  # OS type
                        capabilities.system_type = result
                    elif i == 1:  # Firewall type
                        capabilities.firewall_type = result
                    elif i == 2:  # Package manager
                        capabilities.package_manager = result
                    elif i == 3:  # Security tools
                        capabilities.available_tools = result
                    elif i == 4:  # System info
                        arch, shell, sudo = result
                        capabilities.architecture = arch
                        capabilities.shell_type = shell
                        capabilities.has_sudo = sudo
                    elif i == 5:  # Container environment
                        capabilities.has_docker, capabilities.has_systemctl = result
                        
        except Exception as e:
            self.logger.error(f"System detection failed: {e}")
        
        # Cache the results
        self._capabilities_cache[cache_key] = capabilities
        
        self.logger.info(f"System detection complete: {capabilities.system_type.value}, {capabilities.firewall_type.value}")
        return capabilities
    
    async def _detect_os_type(self) -> SystemType:
        """Detect operating system type"""
        # Check /etc/os-release
        status, stdout, _ = await self.responder.execute_command("cat /etc/os-release", timeout=5)
        if status == "success" and stdout:
            stdout_lower = stdout.lower()
            if "ubuntu" in stdout_lower:
                return SystemType.UBUNTU
            elif "debian" in stdout_lower:
                return SystemType.DEBIAN
            elif "rhel" in stdout_lower or "red hat" in stdout_lower:
                return SystemType.RHEL
            elif "centos" in stdout_lower:
                return SystemType.CENTOS
            elif "alpine" in stdout_lower:
                return SystemType.ALPINE
        
        # Check for T-Pot specific indicators
        status, stdout, _ = await self.responder.execute_command("ls /opt/tpot* /home/*/tpot* 2>/dev/null", timeout=5)
        if status == "success" and "tpot" in stdout.lower():
            return SystemType.TPOT
        
        # Check for Windows
        status, stdout, _ = await self.responder.execute_command("ver", timeout=5)
        if status == "success" and "windows" in stdout.lower():
            return SystemType.WINDOWS
        
        return SystemType.UNKNOWN
    
    async def _detect_firewall_type(self) -> FirewallType:
        """Detect available firewall system"""
        # Check UFW (Ubuntu Firewall)
        status, _, _ = await self.responder.execute_command("which ufw", timeout=5)
        if status == "success":
            # Verify UFW is active
            status, stdout, _ = await self.responder.execute_command("sudo ufw status", timeout=5)
            if status == "success":
                return FirewallType.UFW
        
        # Check firewalld (RHEL/CentOS)
        status, _, _ = await self.responder.execute_command("which firewall-cmd", timeout=5)
        if status == "success":
            return FirewallType.FIREWALLD
        
        # Check iptables (most Linux systems)
        status, _, _ = await self.responder.execute_command("which iptables", timeout=5)
        if status == "success":
            return FirewallType.IPTABLES
        
        # Check Windows Firewall
        status, _, _ = await self.responder.execute_command("netsh advfirewall show allprofiles", timeout=5)
        if status == "success":
            return FirewallType.WINDOWS_FIREWALL
        
        return FirewallType.NONE
    
    async def _detect_package_manager(self) -> str:
        """Detect available package manager"""
        managers = ["apt-get", "yum", "dnf", "pacman", "apk", "brew"]
        
        for manager in managers:
            status, _, _ = await self.responder.execute_command(f"which {manager}", timeout=3)
            if status == "success":
                return manager
        
        return "unknown"
    
    async def _detect_security_tools(self) -> List[str]:
        """Detect available security tools"""
        tools = []
        security_commands = {
            "iptables": "which iptables",
            "ufw": "which ufw", 
            "firewall-cmd": "which firewall-cmd",
            "fail2ban": "which fail2ban-client",
            "docker": "which docker",
            "systemctl": "which systemctl",
            "netstat": "which netstat",
            "ss": "which ss",
            "tcpdump": "which tcpdump",
            "wireshark": "which tshark"
        }
        
        for tool, command in security_commands.items():
            status, _, _ = await self.responder.execute_command(command, timeout=3)
            if status == "success":
                tools.append(tool)
        
        return tools
    
    async def _detect_system_info(self) -> Tuple[str, str, bool]:
        """Detect architecture, shell, and sudo access"""
        # Architecture
        status, arch_stdout, _ = await self.responder.execute_command("uname -m", timeout=3)
        architecture = arch_stdout.strip() if status == "success" else "unknown"
        
        # Shell
        status, shell_stdout, _ = await self.responder.execute_command("echo $SHELL", timeout=3)
        shell_type = shell_stdout.strip().split('/')[-1] if status == "success" else "unknown"
        
        # Sudo access
        status, _, _ = await self.responder.execute_command("sudo -n true", timeout=3)
        has_sudo = (status == "success")
        
        return architecture, shell_type, has_sudo
    
    async def _detect_container_environment(self) -> Tuple[bool, bool]:
        """Detect container and service management capabilities"""
        # Docker
        status, _, _ = await self.responder.execute_command("docker --version", timeout=3)
        has_docker = (status == "success")
        
        # Systemctl
        status, _, _ = await self.responder.execute_command("systemctl --version", timeout=3)
        has_systemctl = (status == "success")
        
        return has_docker, has_systemctl
    
    def generate_adaptive_commands(self, action_type: str, parameters: Dict, capabilities: SystemCapabilities) -> List[str]:
        """
        Generate adaptive commands based on detected system capabilities
        """
        commands = []
        
        if action_type == "block_ip":
            ip = parameters.get("ip_address") or parameters.get("ip")
            duration = parameters.get("duration", "1h")
            
            # Adapt based on firewall type
            if capabilities.firewall_type == FirewallType.UFW:
                commands = [
                    f"sudo ufw deny from {ip}",
                    f"sudo ufw reload"
                ]
                
            elif capabilities.firewall_type == FirewallType.FIREWALLD:
                commands = [
                    f"sudo firewall-cmd --permanent --add-rich-rule='rule source address=\"{ip}\" drop'",
                    f"sudo firewall-cmd --reload"
                ]
                
            elif capabilities.firewall_type == FirewallType.IPTABLES:
                # Check if it's a T-Pot system (more careful with honeypots)
                if capabilities.system_type == SystemType.TPOT:
                    commands = [
                        f"sudo iptables -I INPUT -s {ip} -p tcp --dport 80 -j DROP",
                        f"sudo iptables -I INPUT -s {ip} -p tcp --dport 443 -j DROP",
                        f"sudo iptables -I INPUT -s {ip} -p tcp --dport 2222 -j DROP"
                    ]
                else:
                    commands = [f"sudo iptables -I INPUT -s {ip} -j DROP"]
                    
            elif capabilities.firewall_type == FirewallType.WINDOWS_FIREWALL:
                commands = [
                    f"netsh advfirewall firewall add rule name=\"Block_{ip}\" dir=in action=block remoteip={ip}"
                ]
            else:
                # Fallback: try multiple methods
                commands = [
                    f"sudo iptables -I INPUT -s {ip} -j DROP || echo 'iptables failed'",
                    f"sudo ufw deny from {ip} || echo 'ufw not available'"
                ]
        
        elif action_type == "isolate_host":
            ip = parameters.get("host_identifier") or parameters.get("ip")
            level = parameters.get("isolation_level", "soft")
            
            if capabilities.firewall_type == FirewallType.UFW:
                if level == "hard":
                    commands = [
                        f"sudo ufw deny from {ip}",
                        f"sudo ufw deny to {ip}",
                        f"sudo ufw reload"
                    ]
                else:
                    commands = [
                        f"sudo ufw deny from {ip} to any port 80",
                        f"sudo ufw deny from {ip} to any port 443",
                        f"sudo ufw reload"
                    ]
                    
            elif capabilities.firewall_type == FirewallType.IPTABLES:
                if level == "hard":
                    commands = [
                        f"sudo iptables -I INPUT -s {ip} -j DROP",
                        f"sudo iptables -I OUTPUT -d {ip} -j DROP",
                        f"sudo iptables -I FORWARD -s {ip} -j DROP",
                        f"sudo iptables -I FORWARD -d {ip} -j DROP"
                    ]
                else:
                    # Soft isolation - block web traffic but allow SSH
                    commands = [
                        f"sudo iptables -I INPUT -s {ip} -p tcp --dport 80 -j DROP",
                        f"sudo iptables -I INPUT -s {ip} -p tcp --dport 443 -j DROP"
                    ]
        
        elif action_type == "deploy_firewall_rules":
            rules = parameters.get("rule_set", [])
            
            if capabilities.firewall_type == FirewallType.UFW:
                for rule in rules:
                    commands.append(f"sudo ufw {rule}")
                commands.append("sudo ufw reload")
                
            elif capabilities.firewall_type == FirewallType.FIREWALLD:
                for rule in rules:
                    commands.append(f"sudo firewall-cmd --permanent {rule}")
                commands.append("sudo firewall-cmd --reload")
                
            elif capabilities.firewall_type == FirewallType.IPTABLES:
                commands.extend([f"sudo iptables {rule}" for rule in rules])
        
        return commands
    
    def get_system_summary(self, capabilities: SystemCapabilities) -> str:
        """Generate human-readable system summary"""
        return f"""System: {capabilities.system_type.value}
Firewall: {capabilities.firewall_type.value}
Architecture: {capabilities.architecture}
Shell: {capabilities.shell_type}
Sudo Access: {'Yes' if capabilities.has_sudo else 'No'}
Package Manager: {capabilities.package_manager}
Security Tools: {', '.join(capabilities.available_tools[:5])}
Container Support: {'Docker' if capabilities.has_docker else 'None'}
Service Management: {'systemctl' if capabilities.has_systemctl else 'Legacy'}"""


# Global detector instance
_system_detector = None

async def get_system_detector():
    """Get or create system detector instance"""
    global _system_detector
    if _system_detector is None:
        from .responder import responder
        _system_detector = IntelligentSystemDetector(responder)
    return _system_detector
