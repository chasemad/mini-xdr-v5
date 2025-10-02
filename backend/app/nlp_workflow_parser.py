"""
NLP Workflow Parser for Mini-XDR
Converts natural language descriptions into structured response workflows
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class WorkflowIntent:
    """Represents a parsed workflow intent from natural language"""

    def __init__(self):
        self.primary_goal: Optional[str] = None
        self.target_ip: Optional[str] = None
        self.target_environment: Optional[str] = None  # honeypot, production, staging, etc.
        self.target_system: Optional[str] = None  # specific hostname or system identifier
        self.actions: List[Dict[str, Any]] = []
        self.conditions: Dict[str, Any] = {}
        self.priority: str = "medium"
        self.approval_required: bool = True
        self.confidence: float = 0.0
        self.request_type: str = "response"  # response|investigation|automation|reporting|qa

        # Feedback and recommendations
        self.missing_info: List[str] = []  # What information is needed
        self.recommendations: List[str] = []  # Suggested alternatives
        self.unsupported_actions: List[str] = []  # Actions we can't perform
        self.clarification_needed: bool = False
        self.fallback_used: bool = False

    def to_workflow_steps(self) -> List[Dict[str, Any]]:
        """Convert intent to workflow steps"""
        return self.actions

    def add_recommendation(self, message: str):
        """Add a recommendation for the user"""
        self.recommendations.append(message)

    def add_missing_info(self, info: str):
        """Add required information that's missing"""
        self.missing_info.append(info)
        self.clarification_needed = True

    def add_unsupported_action(self, action: str):
        """Track an action we can't perform"""
        self.unsupported_actions.append(action)

    def get_feedback_message(self) -> str:
        """Generate helpful feedback message for the user"""
        messages = []

        if self.unsupported_actions:
            messages.append(f"⚠️ The following capabilities are not currently available: {', '.join(self.unsupported_actions)}")

        if self.missing_info:
            messages.append(f"❓ I need more information: {', '.join(self.missing_info)}")

        if self.recommendations:
            messages.append(f"💡 Recommendations:\n" + "\n".join(f"  • {r}" for r in self.recommendations))

        if not self.actions and not self.clarification_needed:
            messages.append("❓ I couldn't identify any specific actions from your request. Could you be more specific?")

        return "\n\n".join(messages) if messages else None


class NLPWorkflowParser:
    """
    Intelligent NLP parser that converts natural language into Mini-XDR workflows

    Supports patterns like:
    - "Block IP 192.168.1.100 and isolate the host"
    - "Investigate SSH brute force from 10.0.0.5, then contain if confirmed"
    - "Emergency: Isolate all hosts affected by ransomware and reset passwords"
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key

        # Action mapping from natural language to response actions
        # Comprehensive coverage for all attack types
        self.action_patterns = {
            # Network actions
            r'\b(block|ban|blacklist)\s+(?:ip\s+)?(\d+\.\d+\.\d+\.\d+)': ('block_ip', 'network'),
            r'\b(block|ban)\s+(?:this|the)?\s*(?:attacking\s+)?(?:ip|attacker)': ('block_ip', 'network'),
            r'\b(unblock|whitelist|allow)\s+(?:ip\s+)?(\d+\.\d+\.\d+\.\d+)': ('unblock_ip', 'network'),
            r'\b(deploy|enable|activate)\s+firewall': ('deploy_firewall_rules', 'network'),
            r'\b(deploy|enable|activate)\s+waf': ('deploy_waf_rules', 'cloud'),
            r'\bcapture\s+(?:network\s+)?traffic': ('capture_network_traffic', 'forensics'),
            r'\b(block|stop)\s+(?:c2|command|control)': ('block_c2_traffic', 'network'),

            # Endpoint actions
            r'\b(isolate|quarantine)\s+(?:the\s+)?(?:infected\s+)?(?:compromised\s+)?(?:systems?|hosts?)': ('isolate_host', 'endpoint'),
            r'\b(un-?isolate|restore)\s+(?:the\s+)?host': ('un_isolate_host', 'endpoint'),
            r'\bterminate\s+(?:suspicious\s+)?process(?:es)?': ('terminate_process', 'endpoint'),
            r'\bkill\s+(?:malicious\s+)?process': ('terminate_process', 'endpoint'),
            r'\bhost isolation\b': ('isolate_host_advanced', 'endpoint'),
            r'\bendpoint isolation\b': ('isolate_host_advanced', 'endpoint'),

            # File operations
            r'\b(quarantine|isolate)\s+(?:the\s+)?(?:malicious\s+)?(?:suspicious\s+)?files?': ('isolate_file', 'endpoint'),
            r'\bdelete\s+(?:malicious\s+)?files?': ('delete_malicious_files', 'endpoint'),
            r'\bscan\s+(?:the\s+)?(?:files?|filesystem)': ('scan_filesystem', 'endpoint'),

            # Investigation/Forensics actions
            r'\b(investigate|analyze|examine)\s+': ('investigate_behavior', 'forensics'),
            r'\bcheck\s+(?:database|db)\s+integrity': ('check_database_integrity', 'data'),
            r'\b(hunt|search)\s+(?:for\s+)?similar\s+attacks': ('hunt_similar_attacks', 'forensics'),
            r'\b(threat|intel)\s+lookup': ('threat_intel_lookup', 'forensics'),
            r'\banalyze\s+(?:the\s+)?(?:malware|payload|binary)': ('analyze_malware', 'forensics'),
            r'\bcapture\s+(?:forensic\s+)?evidence': ('capture_forensic_evidence', 'forensics'),
            r'\btrack\s+(?:the\s+)?(?:threat\s+)?actor': ('track_threat_actor', 'forensics'),
            r'\bidentify\s+campaign': ('identify_campaign', 'forensics'),
            r'\bmemory\s+dump(?:ing)?\b': ('memory_dump_collection', 'forensics'),
            r'\bforensic\s+memory\s+(?:capture|collection)\b': ('memory_dump_collection', 'forensics'),
            r'\btraffic\s+(?:analysis|inspection|monitoring)\b': ('capture_network_traffic', 'forensics'),

            # Identity actions
            r'\b(reset|change)\s+password': ('reset_passwords', 'identity'),
            r'\bpassword\s+reset\b': ('reset_passwords', 'identity'),
            r'\brevoke\s+(?:user\s+)?session': ('revoke_user_sessions', 'identity'),
            r'\b(enforce|enable|require)\s+(?:mfa|multi-?factor(?:\s+authentication)?)': ('enforce_mfa', 'identity'),
            r'\bmfa\s+enforcement\b': ('enforce_mfa', 'identity'),
            r'\bdisable\s+(?:the\s+)?(?:compromised\s+)?(?:user\s+)?account': ('disable_user_account', 'identity'),
            r'\bsuspend\s+(?:the\s+)?(?:user|account)': ('disable_user_account', 'identity'),

            # Email actions
            r'\bquarantin(?:e|ing)\s+(?:the\s+)?emails?': ('quarantine_email', 'email'),
            r'\bquarantine\s+email': ('quarantine_email', 'email'),
            r'\bblock(?:ing)?\s+(?:the\s+)?sender(?:\s+domain)?': ('block_sender', 'email'),
            r'\bblock\s+sender': ('block_sender', 'email'),

            # Data protection
            r'\bencrypt\s+(?:sensitive\s+)?data': ('encrypt_sensitive_data', 'data'),
            r'\bbackup\s+(?:critical\s+)?data': ('backup_critical_data', 'data'),
            r'\benable\s+dlp': ('enable_dlp', 'data'),
            r'\bprevent\s+(?:data\s+)?exfiltration': ('enable_dlp', 'data'),
            r'\bcredential\s+stuffing\s+defense\b': ('reset_passwords', 'identity'),

            # Communication/Alerting
            r'\b(send|create)\s+(?:an?\s+)?alert': ('alert_security_analysts', 'communication'),
            r'\b(alert|notify)\s+(?:security\s+)?(?:team|analyst)': ('alert_security_analysts', 'communication'),
            r'\bcreate\s+(?:incident\s+)?(?:response\s+)?case': ('create_incident_case', 'communication'),
            r'\bescalate\s+(?:to\s+)?(?:soc|team)': ('escalate_to_team', 'communication'),
            r'\bnotify\s+': ('alert_security_analysts', 'communication'),

            # Deception/Honeypot
            r'\bdeploy\s+(?:deception\s+)?(?:honeypot|service)': ('deploy_honeypot', 'deception'),
            r'\bactivate\s+honeypot': ('deploy_honeypot', 'deception'),
            
            # Advanced response
            r'\b(mitigate|stop)\s+(?:ddos|dos)': ('deploy_firewall_rules', 'network'),
            r'\b(set\s*up|implement)\s+(?:ddos|dos)\s+protection': ('deploy_firewall_rules', 'network'),
            r'\brate\s+limit(?:ing)?\b': ('api_rate_limiting', 'cloud'),
            r'\bcontain\s+(?:the\s+)?(?:attack|threat)': ('isolate_host', 'endpoint'),
            r'\bprevent\s+(?:lateral\s+)?movement': ('isolate_host', 'endpoint'),

            # Additional network actions
            r'\b(establish|activate|implement)\s+(?:protection|defense|security)': ('deploy_firewall_rules', 'network'),
            r'\bnetwork\s+containment\b': ('isolate_host', 'endpoint'),
            r'\b(sinkhole|redirect)\s+(?:domain|dns|queries)': ('deploy_firewall_rules', 'network'),
            r'\bthrottle\s+(?:bandwidth|traffic|connections)': ('api_rate_limiting', 'cloud'),
            r'\blimit\s+(?:connection|bandwidth|rate)': ('api_rate_limiting', 'cloud'),
            r'\b(deep\s+)?packet\s+inspection\b': ('capture_network_traffic', 'forensics'),
            r'\bmonitor\s+(?:and\s+)?inspect\b': ('capture_network_traffic', 'forensics'),

            # Additional endpoint actions
            r'\b(quarantine|lockdown)\s+(?:the\s+)?(?:workstation|endpoint|system|compromised)': ('isolate_host', 'endpoint'),
            r'\b(stop|end|kill)\s+(?:the\s+)?(?:malware\s+)?process(?:es)?': ('terminate_process', 'endpoint'),
            r'\b(terminate|end)\s+(?:malicious|suspicious)\s+process': ('terminate_process', 'endpoint'),
            r'\bcollect\s+(?:ram|memory)\s+dump\b': ('memory_dump_collection', 'forensics'),

            # Investigation and hunting
            r'\bhunt\s+for\s+(?:ioc|indicators)': ('hunt_similar_attacks', 'forensics'),
            r'\broutine\s+check': ('investigate_behavior', 'forensics'),
            r'\breview\s+(?:user\s+)?access': ('investigate_behavior', 'forensics'),

            # Malware and APT
            r'\b(?:malware|ransomware)\s+(?:infection|detected)': ('isolate_host', 'endpoint'),
            r'\b(?:spread|spreading)': ('isolate_host', 'endpoint'),
            r'\bapt\s+detected': ('investigate_behavior', 'forensics'),
            r'\brequires?\s+(?:urgent\s+)?response': ('alert_security_analysts', 'communication'),
            r'\bneeds?\s+(?:immediate\s+)?containment': ('isolate_host', 'endpoint'),
        }

        # Priority keywords
        self.priority_keywords = {
            'emergency': 'critical',
            'urgent': 'critical',
            'critical': 'critical',
            'high': 'high',
            'important': 'high',
            'normal': 'medium',
            'low': 'low',
            'routine': 'low'
        }

        # Threat type keywords - comprehensive coverage for all honeypot attacks
        self.threat_keywords = {
            # Brute force variants
            'brute force': 'brute_force',
            'password spray': 'password_spray',
            'credential stuffing': 'credential_stuffing',
            'ssh brute': 'ssh_brute_force',
            
            # Malware/Botnet
            'ransomware': 'ransomware',
            'malware': 'malware',
            'botnet': 'botnet',
            'trojan': 'trojan',
            'backdoor': 'backdoor',
            
            # Web attacks
            'phishing': 'phishing',
            'sql injection': 'sql_injection',
            'xss': 'xss',
            'csrf': 'csrf',
            'web attack': 'web_attack',
            
            # Network attacks
            'ddos': 'ddos',
            'dos': 'dos',
            'syn flood': 'syn_flood',
            'udp flood': 'udp_flood',
            
            # Advanced threats
            'apt': 'apt',
            'advanced persistent': 'apt',
            'insider threat': 'insider_threat',
            'lateral movement': 'lateral_movement',
            'privilege escalation': 'privilege_escalation',
            
            # Data/Exfiltration
            'data exfiltration': 'data_exfiltration',
            'data breach': 'data_breach',
            'data theft': 'data_theft',
            
            # Reconnaissance
            'reconnaissance': 'reconnaissance',
            'scanning': 'port_scanning',
            'port scan': 'port_scanning',
            'enumeration': 'enumeration',
            
            # C2 Communication
            'c2': 'command_control',
            'command and control': 'command_control',
            'beaconing': 'c2_beaconing'
        }

        # Threat-driven priority overrides (applied when user doesn't specify priority)
        self.threat_priority_overrides = {
            'ransomware': 'critical',
            'apt': 'critical',
            'command_control': 'critical',
            'malware': 'high',
            'botnet': 'high',
            'ddos': 'high',
            'ssh_brute_force': 'high',
            'credential_stuffing': 'high',
            'insider_threat': 'high'
        }

        self.priority_hierarchy = ['low', 'medium', 'high', 'critical']

        # Request type patterns
        self.request_type_patterns = {
            'automation': [
                r'\bevery\s+time\b',
                r'\bwhenever\b',
                r'\bautomat(?:e|ically)\b',
                r'\balways\s+(?:do|perform|execute)\b',
                r'\bschedule\b',
                r'\bon\s+(?:each|every)\b'
            ],
            'investigation': [
                r'\binvestigat(?:e|ion)\b',
                r'\bsearch\s+for\b',
                r'\bhunt\b',
                r'\bfind\s+(?:out|all)\b',
                r'\banalyze\b',
                r'\bcheck\s+(?:if|for|whether)\b',
                r'\blook\s+(?:up|for)\b',
                r'\bidentify\b'
            ],
            'reporting': [
                r'\breport\b',
                r'\bsummar(?:y|ize)\b',
                r'\bshow\s+(?:me|stats|metrics)\b',
                r'\blist\s+(?:all|recent)\b',
                r'\bexport\b',
                r'\bgenerate\s+report\b'
            ],
            'qa': [
                r'\b(?:what|how|why|when|where|who)\s+(?:is|are|does|do)\b',
                r'\bexplain\b',
                r'\btell\s+me\s+about\b',
                r'\bwhat\'s\b',
                r'\bcan\s+you\s+(?:tell|explain)\b'
            ]
        }

    def _classify_request_type(self, text: str) -> str:
        """Classify the request type based on patterns"""
        text_lower = text.lower()

        # Check each request type
        for request_type, patterns in self.request_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return request_type

        # Default to response if contains action keywords
        action_keywords = ['block', 'isolate', 'quarantine', 'terminate', 'reset', 'deploy', 'enable']
        if any(keyword in text_lower for keyword in action_keywords):
            return 'response'

        # Default to qa for questions
        return 'response'

    async def parse(self, natural_language: str, incident_id: Optional[int] = None, incident_context: Optional[Dict] = None) -> WorkflowIntent:
        """
        Parse natural language into structured workflow intent

        Args:
            natural_language: User's natural language description
            incident_id: Optional incident ID for context
            incident_context: Optional pre-fetched incident context dictionary

        Returns:
            WorkflowIntent object with parsed actions
        """
        intent = WorkflowIntent()
        text_lower = natural_language.lower()

        # If incident context provided, enrich the workflow with context
        if incident_context:
            # Auto-populate target IP if not in natural language
            if not re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', natural_language):
                if incident_context.get('src_ip'):
                    intent.target_ip = incident_context['src_ip']
                    intent.conditions['target_ips'] = [incident_context['src_ip']]
            
            # Use incident threat category if available
            if incident_context.get('threat_category'):
                intent.conditions['threat_category'] = incident_context['threat_category']
            
            # Set priority based on risk score if not explicitly stated
            if incident_context.get('risk_score'):
                risk_score = incident_context['risk_score']
                if risk_score > 0.7 and 'critical' not in text_lower and 'high' not in text_lower:
                    intent.priority = 'high'
                elif risk_score > 0.4 and 'low' not in text_lower:
                    intent.priority = 'medium'
            
            # Add incident context to conditions for action parameters
            intent.conditions['incident_id'] = incident_id
            intent.conditions['incident_context'] = {
                'threat_summary': incident_context.get('threat_summary'),
                'attack_patterns': incident_context.get('attack_patterns', []),
                'escalation_level': incident_context.get('escalation_level'),
                'total_events': incident_context.get('total_events', 0)
            }

        # Classify request type
        intent.request_type = self._classify_request_type(text_lower)

        # Extract priority (may override incident-based priority)
        extracted_priority = self._extract_priority(text_lower)
        if extracted_priority != 'medium':  # If explicitly stated, use it
            intent.priority = extracted_priority

        # Extract threat type
        threat_type = self._extract_threat_type(text_lower)
        if threat_type:
            intent.conditions['threat_type'] = threat_type
            override_priority = self.threat_priority_overrides.get(threat_type)
            if override_priority:
                current_priority = intent.priority if intent.priority in self.priority_hierarchy else 'medium'
                if self.priority_hierarchy.index(override_priority) > self.priority_hierarchy.index(current_priority):
                    intent.priority = override_priority

        # Extract environment/target context
        intent.target_environment = self._extract_environment(text_lower)
        intent.target_system = self._extract_system(text_lower)

        # Extract IP addresses (may override incident IP if explicitly specified)
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        ips = re.findall(ip_pattern, natural_language)
        if ips:
            intent.target_ip = ips[0]
            intent.conditions['target_ips'] = ips

        # Split compound sentences and parse each part
        sentence_parts = self._split_compound_sentence(natural_language)

        # Parse actions from each part using pattern matching
        actions = []
        for part in sentence_parts:
            part_actions = self._extract_actions_pattern_based(part, incident_id)
            actions.extend(part_actions)

        # If we have OpenAI key and no actions found, enhance with AI
        if self.openai_api_key and len(actions) == 0:
            actions = await self._extract_actions_with_ai(natural_language, incident_id)

        # Add environment/system context to all actions
        for action in actions:
            if intent.target_environment:
                action['parameters']['target_environment'] = intent.target_environment
            if intent.target_system:
                action['parameters']['target_system'] = intent.target_system

        intent.actions = actions

        # Detect unsupported capabilities
        unsupported = self._detect_unsupported_capabilities(natural_language.lower())
        for cap in unsupported:
            intent.add_unsupported_action(cap)

        # Provide recommendations if needed
        self._provide_recommendations(intent, natural_language.lower())

        intent.confidence = self._calculate_confidence(intent, natural_language)

        # Determine approval requirement
        intent.approval_required = self._requires_approval(intent)

        logger.info(f"Parsed workflow: {len(intent.actions)} actions, priority={intent.priority}, confidence={intent.confidence:.2f}, feedback={intent.get_feedback_message() is not None}")

        return intent

    def _split_compound_sentence(self, text: str) -> List[str]:
        """
        Split compound sentences into individual actionable parts
        Handles: "do X and Y", "do X then Y", "do X, Y, and Z"
        """
        # Split on common separators while preserving the original text
        separators = [' and ', ', and ', ' then ', '; ', ', ']
        parts = [text]

        for separator in separators:
            new_parts = []
            for part in parts:
                if separator in part.lower():
                    split_parts = re.split(re.escape(separator), part, flags=re.IGNORECASE)
                    new_parts.extend([p.strip() for p in split_parts if p.strip()])
                else:
                    new_parts.append(part)
            parts = new_parts

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            if part.lower() not in seen:
                seen.add(part.lower())
                unique_parts.append(part)

        return unique_parts

    def _extract_environment(self, text: str) -> Optional[str]:
        """Extract target environment from text"""
        environment_patterns = {
            r'\bon\s+(?:the\s+)?honeypot': 'honeypot',
            r'\bin\s+(?:the\s+)?honeypot': 'honeypot',
            r'\bon\s+(?:the\s+)?production': 'production',
            r'\bin\s+production': 'production',
            r'\bon\s+staging': 'staging',
            r'\bin\s+staging': 'staging',
            r'\bon\s+(?:the\s+)?dev(?:elopment)?': 'development',
            r'\bfirewall': 'firewall',
        }

        for pattern, environment in environment_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return environment

        return None

    def _extract_system(self, text: str) -> Optional[str]:
        """Extract specific system/host identifier from text"""
        # Look for hostnames like "web-server-01", "db-prod-1", etc.
        system_pattern = r'\bon\s+([a-z0-9\-]+\-[a-z0-9\-]+)'
        match = re.search(system_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_priority(self, text: str) -> str:
        """Extract priority from text"""
        for keyword, priority in self.priority_keywords.items():
            if keyword in text:
                return priority
        return "medium"

    def _extract_threat_type(self, text: str) -> Optional[str]:
        """Extract threat type from text"""
        for keyword, threat_type in self.threat_keywords.items():
            if keyword in text:
                return threat_type
        return None

    def _detect_unsupported_capabilities(self, text: str) -> List[str]:
        """Detect if user is requesting capabilities we don't have"""
        unsupported = []

        # Check for common unsupported requests
        unsupported_patterns = {
            r'\b(uninstall|remove)\s+(?:software|application|program)': 'Software uninstallation (use manual process)',
            r'\b(deploy|install)\s+(?:agent|sensor|edr)': 'Agent deployment (use deployment tool)',
            r'\b(patch|update)\s+(?:system|os|software)': 'System patching (use patch management system)',
            r'\b(physical|hardware)\s+(?:access|security)': 'Physical security (not digital response)',
            r'\b(hire|fire|recruit)\b': 'HR actions (not security actions)',
            r'\b(purchase|buy)\s+': 'Procurement (not security action)',
            r'\b(decompile|reverse\s+engineer)\b': 'Advanced malware analysis (use manual analysis)',
            r'\b(decrypt|crack)\s+(?:password|encryption)': 'Password cracking (use password reset instead)',
            r'\b(social\s+engineer|phish)\s+(?:back|them)': 'Offensive operations (defensive only)',
            r'\b(hack|exploit|attack)\s+(?:back|them)': 'Offensive operations (defensive only)',
            r'\b(reboot|restart)\s+(?:all|every|entire)': 'Mass system reboot (requires manual approval)',
            r'\b(format|wipe|erase)\s+(?:disk|drive|system)': 'Data destruction (requires manual approval)',
        }

        for pattern, capability in unsupported_patterns.items():
            if re.search(pattern, text):
                unsupported.append(capability)

        return unsupported

    def _provide_recommendations(self, intent: WorkflowIntent, text: str):
        """Provide helpful recommendations based on the request"""

        # If no actions found, suggest alternatives
        if not intent.actions:
            if any(word in text for word in ['block', 'ban', 'stop']):
                intent.add_recommendation("Try: 'Block IP [address]' or 'Block sender [email]'")

            if any(word in text for word in ['isolate', 'quarantine', 'contain']):
                intent.add_recommendation("Try: 'Isolate host [hostname]' or 'Quarantine malicious files'")

            if any(word in text for word in ['investigate', 'analyze', 'examine']):
                intent.add_recommendation("Try: 'Investigate malware infection' or 'Analyze threat from [IP]'")

            if any(word in text for word in ['alert', 'notify', 'inform']):
                intent.add_recommendation("Try: 'Alert security analysts about [threat]'")

        # If missing critical information
        if intent.actions:
            for action in intent.actions:
                action_type = action.get('action_type', '')

                if 'block_ip' in action_type and 'ip_address' not in action.get('parameters', {}):
                    intent.add_missing_info("IP address to block")

                if 'isolate_host' in action_type and not any(k in action.get('parameters', {}) for k in ['hostname', 'host_identifier']):
                    intent.add_missing_info("Hostname or system identifier to isolate")

                if 'reset_passwords' in action_type:
                    intent.add_recommendation("Password resets require user communication - consider adding notification action")

        # Priority recommendations
        if intent.priority == 'critical':
            intent.add_recommendation("Critical priority actions require approval. Set auto_execute=false initially.")

        # Request type specific recommendations
        if intent.request_type == 'automation':
            intent.add_recommendation("Automation triggers should include clear conditions (e.g., 'when event_type=brute_force')")

        if intent.request_type == 'qa' and not intent.actions:
            intent.add_recommendation("For questions, I can help explain security concepts or system capabilities")

        if intent.request_type == 'reporting' and not intent.actions:
            intent.add_recommendation("Reporting requests don't create automated workflows - they query existing data")

    def _extract_actions_pattern_based(self, text: str, incident_id: Optional[int]) -> List[Dict[str, Any]]:
        """Extract actions using regex patterns"""
        actions = []
        text_lower = text.lower()

        for pattern, (action_type, category) in self.action_patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract any captured groups (like IP addresses)
                params = {
                    'reason': f'NLP workflow: {text[:100]}',
                    'executed_by': 'nlp_parser'
                }

                if incident_id:
                    params['incident_id'] = incident_id

                # Add extracted values from regex groups
                if match.groups():
                    for i, group in enumerate(match.groups()):
                        if group:
                            if '.' in group and group.replace('.', '').isdigit():
                                params['ip_address'] = group
                            else:
                                params[f'param_{i}'] = group

                actions.append({
                    'action_type': action_type,
                    'category': category,
                    'parameters': params,
                    'timeout_seconds': 300,
                    'continue_on_failure': False,
                    'max_retries': 3
                })

        return self._deduplicate_actions(actions)

    async def _extract_actions_with_ai(self, text: str, incident_id: Optional[int]) -> List[Dict[str, Any]]:
        """Use OpenAI to extract actions from ambiguous text"""
        try:
            import openai
            openai.api_key = self.openai_api_key

            prompt = f"""You are a cybersecurity response automation system. Convert this natural language request into a structured list of response actions.

User request: "{text}"

Available action types:
- block_ip, unblock_ip, deploy_firewall_rules
- isolate_host, un_isolate_host, terminate_process
- reset_passwords, revoke_user_sessions, disable_user_account
- threat_intel_lookup, hunt_similar_attacks, investigate_behavior
- quarantine_email, block_sender
- alert_security_analysts, create_incident_case

Respond in JSON format:
{{"actions": [{{"action_type": "...", "category": "...", "reason": "..."}}]}}
"""

            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content
            import json
            data = json.loads(result)

            # Convert to workflow format
            actions = []
            for action_data in data.get('actions', []):
                actions.append({
                    'action_type': action_data['action_type'],
                    'category': action_data.get('category', 'other'),
                    'parameters': {
                        'reason': action_data.get('reason', text[:100]),
                        'incident_id': incident_id,
                        'executed_by': 'nlp_ai_parser'
                    },
                    'timeout_seconds': 300,
                    'continue_on_failure': False,
                    'max_retries': 3
                })

            return actions

        except Exception as e:
            logger.error(f"AI-based action extraction failed: {e}")
            return []

    def _deduplicate_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate actions"""
        seen = set()
        unique_actions = []

        for action in actions:
            key = (action['action_type'], action['category'])
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        return unique_actions

    def _calculate_confidence(self, intent: WorkflowIntent, original_text: str) -> float:
        """Calculate confidence score for parsed intent"""
        confidence = 0.0

        # Base confidence from number of actions found
        if len(intent.actions) > 0:
            confidence += 0.4

        # Bonus for clear action keywords
        action_keywords = ['block', 'isolate', 'investigate', 'alert', 'quarantine', 'reset']
        for keyword in action_keywords:
            if keyword in original_text.lower():
                confidence += 0.1

        # Bonus for specific targets (IPs, etc.)
        if intent.target_ip:
            confidence += 0.2

        # Bonus for priority indication
        if intent.priority != 'medium':
            confidence += 0.1

        return min(confidence, 1.0)

    def _requires_approval(self, intent: WorkflowIntent) -> bool:
        """Determine if workflow requires human approval"""
        # Critical priority always requires approval
        if intent.priority == 'critical':
            return True

        # Destructive actions require approval
        destructive_actions = ['terminate_process', 'disable_user_account',
                              'encrypt_sensitive_data', 'delete_malicious_files',
                              'isolate_host', 'isolate_host_advanced',
                              'reset_passwords', 'deploy_firewall_rules',
                              'api_rate_limiting']

        for action in intent.actions:
            if action['action_type'] in destructive_actions:
                return True

        # More than 5 actions requires approval
        if len(intent.actions) > 5:
            return True

        return False

    def generate_explanation(self, intent: WorkflowIntent, original_text: str) -> str:
        """Generate human-readable explanation of parsed workflow"""
        lines = [
            f"📝 **Parsed Workflow from:** \"{original_text}\"",
            "",
            f"**Priority:** {intent.priority.upper()}",
            f"**Actions Identified:** {len(intent.actions)}",
            f"**Confidence:** {intent.confidence * 100:.0f}%",
            f"**Approval Required:** {'Yes' if intent.approval_required else 'No'}",
            ""
        ]

        # Add context information
        context_info = []
        if intent.target_ip:
            context_info.append(f"**Target IP:** {intent.target_ip}")
        if intent.target_environment:
            context_info.append(f"**Target Environment:** {intent.target_environment}")
        if intent.target_system:
            context_info.append(f"**Target System:** {intent.target_system}")

        if context_info:
            lines.extend(context_info)
            lines.append("")

        if intent.actions:
            lines.append("**Workflow Steps:**")
            for i, action in enumerate(intent.actions, 1):
                action_desc = f"{i}. {action['action_type'].replace('_', ' ').title()} ({action['category']})"

                # Add environment context if present
                params = action.get('parameters', {})
                if 'target_environment' in params:
                    action_desc += f" on {params['target_environment']}"
                elif 'target_system' in params:
                    action_desc += f" on {params['target_system']}"

                lines.append(action_desc)
        else:
            lines.append("⚠️ No actions could be identified. Please rephrase your request or use the manual designer.")

        return "\n".join(lines)


# Global parser instance
_parser_instance: Optional[NLPWorkflowParser] = None

def get_nlp_parser() -> NLPWorkflowParser:
    """Get or create global NLP parser instance"""
    global _parser_instance
    if _parser_instance is None:
        from .config import settings
        _parser_instance = NLPWorkflowParser(openai_api_key=settings.openai_api_key)
    return _parser_instance


# Convenience function
async def parse_workflow_from_natural_language(
    text: str,
    incident_id: Optional[int] = None,
    db_session = None
) -> Tuple[WorkflowIntent, str]:
    """
    Parse natural language into workflow intent and explanation
    
    Automatically fetches incident context if incident_id is provided

    Returns:
        (WorkflowIntent, explanation_text)
    """
    parser = get_nlp_parser()
    
    # Fetch incident context if incident_id provided
    incident_context = None
    if incident_id and db_session:
        try:
            from .models import Incident, Event, Action
            from sqlalchemy import select
            
            # Get incident
            stmt = select(Incident).where(Incident.id == incident_id)
            result = await db_session.execute(stmt)
            incident = result.scalar_one_or_none()
            
            if incident:
                # Get recent events for context
                events_query = select(Event).where(
                    Event.src_ip == incident.src_ip
                ).order_by(Event.ts.desc()).limit(50)
                events_result = await db_session.execute(events_query)
                events = events_result.scalars().all()
                
                # Build simplified context
                attack_patterns = []
                if incident.reason:
                    if "brute" in incident.reason.lower() or "ssh" in incident.reason.lower():
                        attack_patterns.append("SSH brute-force")
                    if "sql" in incident.reason.lower():
                        attack_patterns.append("SQL injection")
                    if "malware" in incident.reason.lower():
                        attack_patterns.append("Malware delivery")
                
                incident_context = {
                    'src_ip': incident.src_ip,
                    'threat_summary': incident.reason,
                    'risk_score': incident.risk_score or 0.0,
                    'escalation_level': incident.escalation_level or 'medium',
                    'threat_category': incident.threat_category,
                    'attack_patterns': attack_patterns or ['Unknown pattern'],
                    'total_events': len(events)
                }
        except Exception as e:
            logger.warning(f"Failed to fetch incident context for NLP parsing: {e}")
    
    intent = await parser.parse(text, incident_id, incident_context)
    explanation = parser.generate_explanation(intent, text)
    return intent, explanation
