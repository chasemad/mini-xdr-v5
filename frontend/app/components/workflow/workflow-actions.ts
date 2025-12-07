import {
    Shield,
    Network,
    Server,
    Mail,
    Cloud,
    Key,
    Database,
    Zap,
    Target,
    Bot,
    Webhook,
    Clock,
    AlertCircle,
    Globe,
    Lock,
    FileSearch,
    Terminal,
    Search
} from 'lucide-react';

export type ActionCategory = 'trigger' | 'agent' | 'app' | 'core' | 'network' | 'endpoint' | 'email' | 'identity' | 'security';

export interface WorkflowAction {
    id: string;
    name: string;
    description: string;
    category: ActionCategory;
    subcategory?: string; // e.g., "CrowdStrike", "Okta"
    icon?: any;
    safety_level: 'low' | 'medium' | 'high';
    estimated_duration: number; // in seconds
    rollback_supported: boolean;
    inputs?: Record<string, any>;
}

export const WORKFLOW_ACTIONS: Record<string, WorkflowAction> = {
    // ==================== TRIGGERS ====================
    // Core Triggers
    'trigger_manual': {
        id: 'trigger_manual',
        name: 'Manual Trigger',
        description: 'Manually start workflow on demand',
        category: 'trigger',
        subcategory: 'Core',
        icon: Zap,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'trigger_webhook': {
        id: 'trigger_webhook',
        name: 'Webhook',
        description: 'Start workflow on incoming HTTP request',
        category: 'trigger',
        subcategory: 'Core',
        icon: Webhook,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'trigger_schedule': {
        id: 'trigger_schedule',
        name: 'Schedule',
        description: 'Run workflow at fixed intervals or cron schedule',
        category: 'trigger',
        subcategory: 'Core',
        icon: Clock,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },

    // SIEM Triggers
    'trigger_splunk': {
        id: 'trigger_splunk',
        name: 'Splunk Alert',
        description: 'Trigger on specific Splunk search results',
        category: 'trigger',
        subcategory: 'Splunk',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'trigger_elastic': {
        id: 'trigger_elastic',
        name: 'Elastic Alert',
        description: 'Trigger on Elasticsearch/Kibana alerts',
        category: 'trigger',
        subcategory: 'Elastic',
        icon: Database,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'trigger_sentinel': {
        id: 'trigger_sentinel',
        name: 'Azure Sentinel',
        description: 'Trigger on Azure Sentinel incidents',
        category: 'trigger',
        subcategory: 'Microsoft',
        icon: Shield,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },

    // EDR Triggers
    'trigger_crowdstrike': {
        id: 'trigger_crowdstrike',
        name: 'CrowdStrike Detection',
        description: 'Trigger on new endpoint detection',
        category: 'trigger',
        subcategory: 'CrowdStrike',
        icon: Shield,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'trigger_sentinelone': {
        id: 'trigger_sentinelone',
        name: 'SentinelOne Alert',
        description: 'Trigger on SentinelOne threat detection',
        category: 'trigger',
        subcategory: 'SentinelOne',
        icon: Shield,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },

    // ==================== AI AGENTS ====================
    'agent_triage': {
        id: 'agent_triage',
        name: 'Triage Agent',
        description: 'Analyzes alert context to determine severity and false positives',
        category: 'agent',
        subcategory: 'Investigation',
        icon: Bot,
        safety_level: 'low',
        estimated_duration: 30,
        rollback_supported: false
    },
    'agent_malware': {
        id: 'agent_malware',
        name: 'Malware Analyst',
        description: 'Deep analysis of suspicious files and hashes',
        category: 'agent',
        subcategory: 'Forensics',
        icon: FileSearch,
        safety_level: 'low',
        estimated_duration: 120,
        rollback_supported: false
    },
    'agent_phishing': {
        id: 'agent_phishing',
        name: 'Phishing Responder',
        description: 'Investigates email headers, links, and attachments',
        category: 'agent',
        subcategory: 'Email',
        icon: Mail,
        safety_level: 'low',
        estimated_duration: 45,
        rollback_supported: false
    },
    'agent_threat_hunter': {
        id: 'agent_threat_hunter',
        name: 'Threat Hunter',
        description: 'Proactive threat hunting and IOC correlation',
        category: 'agent',
        subcategory: 'Investigation',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 180,
        rollback_supported: false
    },
    'agent_forensics': {
        id: 'agent_forensics',
        name: 'Forensics Investigator',
        description: 'Deep dive forensic analysis of systems and incidents',
        category: 'agent',
        subcategory: 'Forensics',
        icon: FileSearch,
        safety_level: 'low',
        estimated_duration: 300,
        rollback_supported: false
    },
    'agent_compliance': {
        id: 'agent_compliance',
        name: 'Compliance Auditor',
        description: 'Checks actions against compliance requirements',
        category: 'agent',
        subcategory: 'Compliance',
        icon: Shield,
        safety_level: 'low',
        estimated_duration: 20,
        rollback_supported: false
    },
    'agent_ioc_extractor': {
        id: 'agent_ioc_extractor',
        name: 'IOC Extractor',
        description: 'Extracts indicators of compromise from logs and artifacts',
        category: 'agent',
        subcategory: 'Investigation',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 15,
        rollback_supported: false
    },

    // ==================== EDR/XDR INTEGRATIONS ====================
    // CrowdStrike
    'cs_isolate': {
        id: 'cs_isolate',
        name: 'Isolate Host',
        description: 'Network contain an endpoint via CrowdStrike',
        category: 'app',
        subcategory: 'CrowdStrike',
        icon: Shield,
        safety_level: 'high',
        estimated_duration: 10,
        rollback_supported: true
    },
    'cs_lift_containment': {
        id: 'cs_lift_containment',
        name: 'Lift Containment',
        description: 'Remove network containment from host',
        category: 'app',
        subcategory: 'CrowdStrike',
        icon: Shield,
        safety_level: 'medium',
        estimated_duration: 10,
        rollback_supported: false
    },
    'cs_get_processes': {
        id: 'cs_get_processes',
        name: 'Get Processes',
        description: 'List running processes on host',
        category: 'app',
        subcategory: 'CrowdStrike',
        icon: Terminal,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },
    'cs_kill_process': {
        id: 'cs_kill_process',
        name: 'Kill Process',
        description: 'Terminate a specific process on endpoint',
        category: 'app',
        subcategory: 'CrowdStrike',
        icon: Terminal,
        safety_level: 'high',
        estimated_duration: 5,
        rollback_supported: false
    },
    'cs_get_detections': {
        id: 'cs_get_detections',
        name: 'Get Detections',
        description: 'Retrieve recent detections for a host',
        category: 'app',
        subcategory: 'CrowdStrike',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // SentinelOne
    's1_isolate': {
        id: 's1_isolate',
        name: 'Isolate Endpoint',
        description: 'Disconnect endpoint from network',
        category: 'app',
        subcategory: 'SentinelOne',
        icon: Shield,
        safety_level: 'high',
        estimated_duration: 10,
        rollback_supported: true
    },
    's1_remediate': {
        id: 's1_remediate',
        name: 'Auto Remediate',
        description: 'Execute threat remediation actions',
        category: 'app',
        subcategory: 'SentinelOne',
        icon: Target,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: true
    },

    // Microsoft Defender
    'defender_isolate': {
        id: 'defender_isolate',
        name: 'Isolate Device',
        description: 'Isolate device via Microsoft Defender',
        category: 'app',
        subcategory: 'Microsoft Defender',
        icon: Shield,
        safety_level: 'high',
        estimated_duration: 15,
        rollback_supported: true
    },
    'defender_scan': {
        id: 'defender_scan',
        name: 'Run Antivirus Scan',
        description: 'Initiate full antivirus scan',
        category: 'app',
        subcategory: 'Microsoft Defender',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 300,
        rollback_supported: false
    },

    // ==================== IAM/IDENTITY ====================
    // Okta
    'okta_suspend': {
        id: 'okta_suspend',
        name: 'Suspend User',
        description: 'Suspend a user account in Okta',
        category: 'app',
        subcategory: 'Okta',
        icon: Lock,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: true
    },
    'okta_unsuspend': {
        id: 'okta_unsuspend',
        name: 'Unsuspend User',
        description: 'Reactivate suspended user account',
        category: 'app',
        subcategory: 'Okta',
        icon: Key,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false
    },
    'okta_reset_pwd': {
        id: 'okta_reset_pwd',
        name: 'Reset Password',
        description: 'Trigger password reset flow',
        category: 'app',
        subcategory: 'Okta',
        icon: Key,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false
    },
    'okta_revoke_sessions': {
        id: 'okta_revoke_sessions',
        name: 'Revoke Sessions',
        description: 'Invalidate all active user sessions',
        category: 'app',
        subcategory: 'Okta',
        icon: Lock,
        safety_level: 'high',
        estimated_duration: 5,
        rollback_supported: false
    },
    'okta_force_mfa': {
        id: 'okta_force_mfa',
        name: 'Force MFA Re-enrollment',
        description: 'Require user to re-enroll MFA factors',
        category: 'app',
        subcategory: 'Okta',
        icon: Shield,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Azure AD
    'azuread_disable': {
        id: 'azuread_disable',
        name: 'Disable Account',
        description: 'Disable user account in Azure AD',
        category: 'app',
        subcategory: 'Azure AD',
        icon: Lock,
        safety_level: 'high',
        estimated_duration: 5,
        rollback_supported: true
    },
    'azuread_revoke_tokens': {
        id: 'azuread_revoke_tokens',
        name: 'Revoke Refresh Tokens',
        description: 'Invalidate all refresh tokens for user',
        category: 'app',
        subcategory: 'Azure AD',
        icon: Key,
        safety_level: 'high',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Active Directory
    'ad_disable_user': {
        id: 'ad_disable_user',
        name: 'Disable User',
        description: 'Disable Active Directory user account',
        category: 'app',
        subcategory: 'Active Directory',
        icon: Lock,
        safety_level: 'high',
        estimated_duration: 5,
        rollback_supported: true
    },
    'ad_remove_from_group': {
        id: 'ad_remove_from_group',
        name: 'Remove from Group',
        description: 'Remove user from security groups',
        category: 'app',
        subcategory: 'Active Directory',
        icon: Key,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: true
    },

    // ==================== THREAT INTELLIGENCE ====================
    // VirusTotal
    'vt_scan_ip': {
        id: 'vt_scan_ip',
        name: 'Scan IP',
        description: 'Check IP reputation on VirusTotal',
        category: 'app',
        subcategory: 'VirusTotal',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },
    'vt_scan_domain': {
        id: 'vt_scan_domain',
        name: 'Scan Domain',
        description: 'Check domain reputation',
        category: 'app',
        subcategory: 'VirusTotal',
        icon: Globe,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },
    'vt_scan_hash': {
        id: 'vt_scan_hash',
        name: 'Scan File Hash',
        description: 'Look up file hash reputation',
        category: 'app',
        subcategory: 'VirusTotal',
        icon: FileSearch,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // ThreatConnect
    'tc_lookup_indicator': {
        id: 'tc_lookup_indicator',
        name: 'Lookup Indicator',
        description: 'Query ThreatConnect for IOC intelligence',
        category: 'app',
        subcategory: 'ThreatConnect',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // MISP
    'misp_search': {
        id: 'misp_search',
        name: 'Search Events',
        description: 'Search MISP for related threat events',
        category: 'app',
        subcategory: 'MISP',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 10,
        rollback_supported: false
    },
    'misp_add_attribute': {
        id: 'misp_add_attribute',
        name: 'Add Attribute',
        description: 'Add IOC attribute to MISP',
        category: 'app',
        subcategory: 'MISP',
        icon: Database,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // ==================== EMAIL SECURITY ====================
    // Office 365
    'o365_delete_email': {
        id: 'o365_delete_email',
        name: 'Delete Email',
        description: 'Remove email from user mailboxes',
        category: 'app',
        subcategory: 'Office 365',
        icon: Mail,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: false
    },
    'o365_get_email': {
        id: 'o365_get_email',
        name: 'Get Email Details',
        description: 'Retrieve email metadata and headers',
        category: 'app',
        subcategory: 'Office 365',
        icon: Mail,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Proofpoint
    'pp_quarantine': {
        id: 'pp_quarantine',
        name: 'Quarantine Email',
        description: 'Move email to quarantine',
        category: 'app',
        subcategory: 'Proofpoint',
        icon: Mail,
        safety_level: 'medium',
        estimated_duration: 10,
        rollback_supported: true
    },
    'pp_block_sender': {
        id: 'pp_block_sender',
        name: 'Block Sender',
        description: 'Add sender to block list',
        category: 'app',
        subcategory: 'Proofpoint',
        icon: Lock,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: true
    },

    // ==================== NETWORK/FIREWALL ====================
    // Palo Alto
    'palo_block_ip': {
        id: 'palo_block_ip',
        name: 'Block IP',
        description: 'Add IP to deny list on Palo Alto firewall',
        category: 'app',
        subcategory: 'Palo Alto',
        icon: Network,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: true
    },
    'palo_block_domain': {
        id: 'palo_block_domain',
        name: 'Block Domain',
        description: 'Block domain at firewall',
        category: 'app',
        subcategory: 'Palo Alto',
        icon: Globe,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: true
    },

    // Cisco
    'cisco_block_ip': {
        id: 'cisco_block_ip',
        name: 'Block IP',
        description: 'Add IP to ACL deny list',
        category: 'app',
        subcategory: 'Cisco',
        icon: Network,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: true
    },

    // ==================== CLOUD SECURITY ====================
    // AWS
    'aws_revoke_keys': {
        id: 'aws_revoke_keys',
        name: 'Revoke API Keys',
        description: 'Deactivate compromised AWS access keys',
        category: 'app',
        subcategory: 'AWS',
        icon: Cloud,
        safety_level: 'high',
        estimated_duration: 10,
        rollback_supported: false
    },
    'aws_snapshot_ec2': {
        id: 'aws_snapshot_ec2',
        name: 'Snapshot EC2',
        description: 'Create forensic snapshot of EC2 instance',
        category: 'app',
        subcategory: 'AWS',
        icon: Database,
        safety_level: 'low',
        estimated_duration: 60,
        rollback_supported: false
    },
    'aws_isolate_sg': {
        id: 'aws_isolate_sg',
        name: 'Isolate Security Group',
        description: 'Apply restrictive security group',
        category: 'app',
        subcategory: 'AWS',
        icon: Lock,
        safety_level: 'high',
        estimated_duration: 10,
        rollback_supported: true
    },

    // Azure
    'azure_disable_vm': {
        id: 'azure_disable_vm',
        name: 'Stop VM',
        description: 'Shutdown Azure virtual machine',
        category: 'app',
        subcategory: 'Azure',
        icon: Server,
        safety_level: 'high',
        estimated_duration: 30,
        rollback_supported: true
    },

    // ==================== TICKETING/COMMUNICATION ====================
    // ServiceNow
    'snow_create_incident': {
        id: 'snow_create_incident',
        name: 'Create Incident',
        description: 'Create ServiceNow incident ticket',
        category: 'app',
        subcategory: 'ServiceNow',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 10,
        rollback_supported: false
    },
    'snow_update_incident': {
        id: 'snow_update_incident',
        name: 'Update Incident',
        description: 'Update existing incident with notes',
        category: 'app',
        subcategory: 'ServiceNow',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Jira
    'jira_create_ticket': {
        id: 'jira_create_ticket',
        name: 'Create Ticket',
        description: 'Create Jira ticket for tracking',
        category: 'app',
        subcategory: 'Jira',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Slack
    'slack_message': {
        id: 'slack_message',
        name: 'Send Message',
        description: 'Post a message to a Slack channel',
        category: 'app',
        subcategory: 'Slack',
        icon: Globe,
        safety_level: 'low',
        estimated_duration: 2,
        rollback_supported: false
    },
    'slack_notify_oncall': {
        id: 'slack_notify_oncall',
        name: 'Notify On-Call',
        description: 'Alert on-call engineer via Slack',
        category: 'app',
        subcategory: 'Slack',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 2,
        rollback_supported: false
    },

    // Microsoft Teams
    'teams_message': {
        id: 'teams_message',
        name: 'Send Message',
        description: 'Post message to Teams channel',
        category: 'app',
        subcategory: 'Microsoft Teams',
        icon: Globe,
        safety_level: 'low',
        estimated_duration: 2,
        rollback_supported: false
    },

    // PagerDuty
    'pd_create_incident': {
        id: 'pd_create_incident',
        name: 'Create Incident',
        description: 'Trigger PagerDuty incident',
        category: 'app',
        subcategory: 'PagerDuty',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Email
    'send_email': {
        id: 'send_email',
        name: 'Send Email',
        description: 'Send notification email',
        category: 'app',
        subcategory: 'Communication',
        icon: Mail,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // ==================== SIEM/LOG ANALYSIS ====================
    // Splunk
    'splunk_search': {
        id: 'splunk_search',
        name: 'Run Search',
        description: 'Execute Splunk search query',
        category: 'app',
        subcategory: 'Splunk',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 30,
        rollback_supported: false
    },
    'splunk_add_notable': {
        id: 'splunk_add_notable',
        name: 'Create Notable Event',
        description: 'Add notable event to Enterprise Security',
        category: 'app',
        subcategory: 'Splunk',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 5,
        rollback_supported: false
    },

    // Elastic
    'elastic_query': {
        id: 'elastic_query',
        name: 'Query Logs',
        description: 'Execute Elasticsearch query',
        category: 'app',
        subcategory: 'Elastic',
        icon: Search,
        safety_level: 'low',
        estimated_duration: 15,
        rollback_supported: false
    },

    // ==================== SANDBOX/ANALYSIS ====================
    'anyrun_submit': {
        id: 'anyrun_submit',
        name: 'Submit to Any.Run',
        description: 'Detonate file in Any.Run sandbox',
        category: 'app',
        subcategory: 'Any.Run',
        icon: FileSearch,
        safety_level: 'low',
        estimated_duration: 120,
        rollback_supported: false
    },

    // ==================== CORE LOGIC ====================
    'core_if': {
        id: 'core_if',
        name: 'If Condition',
        description: 'Execute different paths based on condition',
        category: 'core',
        subcategory: 'Logic',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_switch': {
        id: 'core_switch',
        name: 'Switch',
        description: 'Route to different paths based on value',
        category: 'core',
        subcategory: 'Logic',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_loop': {
        id: 'core_loop',
        name: 'Loop',
        description: 'Iterate over items in array',
        category: 'core',
        subcategory: 'Logic',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_delay': {
        id: 'core_delay',
        name: 'Wait',
        description: 'Pause workflow for a set duration',
        category: 'core',
        subcategory: 'Utility',
        icon: Clock,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_http': {
        id: 'core_http',
        name: 'HTTP Request',
        description: 'Make a generic HTTP request to any API',
        category: 'core',
        subcategory: 'Network',
        icon: Globe,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false
    },
    'core_webhook': {
        id: 'core_webhook',
        name: 'Webhook',
        description: 'Send data to external webhook URL',
        category: 'core',
        subcategory: 'Network',
        icon: Webhook,
        safety_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false
    },
    'core_code': {
        id: 'core_code',
        name: 'Run Code',
        description: 'Execute Python or JavaScript code',
        category: 'core',
        subcategory: 'Advanced',
        icon: Terminal,
        safety_level: 'high',
        estimated_duration: 10,
        rollback_supported: false
    },
    'core_transform': {
        id: 'core_transform',
        name: 'Transform Data',
        description: 'Map, filter, or transform data',
        category: 'core',
        subcategory: 'Data',
        icon: Database,
        safety_level: 'low',
        estimated_duration: 1,
        rollback_supported: false
    },
    'core_merge': {
        id: 'core_merge',
        name: 'Merge',
        description: 'Combine data from multiple branches',
        category: 'core',
        subcategory: 'Logic',
        icon: Target,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_error_handler': {
        id: 'core_error_handler',
        name: 'Error Handler',
        description: 'Handle errors and exceptions gracefully',
        category: 'core',
        subcategory: 'Logic',
        icon: AlertCircle,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
    'core_stop': {
        id: 'core_stop',
        name: 'Stop Workflow',
        description: 'Terminate workflow execution',
        category: 'core',
        subcategory: 'Logic',
        icon: Lock,
        safety_level: 'low',
        estimated_duration: 0,
        rollback_supported: false
    },
};
