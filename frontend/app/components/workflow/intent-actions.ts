/**
 * Intent-Based Workflow Actions
 *
 * Actions organized by SECURITY OBJECTIVE rather than vendor.
 * AI intelligently selects vendor implementation at runtime.
 */

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
    Search,
    Ban,
    UserX,
    HardDrive,
    Trash2,
    MessageSquare,
    Play,
    Pause,
    RefreshCw
} from 'lucide-react';

export type SecurityPhase =
    | 'trigger'
    | 'investigation'
    | 'containment'
    | 'access_control'
    | 'forensics'
    | 'remediation'
    | 'communication'
    | 'core';

export interface IntentAction {
    id: string;
    name: string;
    description: string;
    phase: SecurityPhase;
    icon: any;

    // Risk and execution metadata
    risk_level: 'low' | 'medium' | 'high' | 'critical';
    estimated_duration: number; // seconds
    rollback_supported: boolean;
    requires_approval: boolean;

    // AI orchestration metadata
    intent_type: string; // e.g., "block_ip", "isolate_host"
    vendor_agnostic: boolean; // If true, AI selects vendor

    // Optional parameters schema
    parameters?: {
        required: string[];
        optional: string[];
        schema: Record<string, any>;
    };
}

export const INTENT_BASED_ACTIONS: Record<string, IntentAction> = {
    // ==================== TRIGGERS ====================
    'trigger_manual': {
        id: 'trigger_manual',
        name: 'Manual Trigger',
        description: 'Manually start workflow on demand',
        phase: 'trigger',
        icon: Zap,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'trigger_manual',
        vendor_agnostic: false
    },
    'trigger_webhook': {
        id: 'trigger_webhook',
        name: 'Webhook',
        description: 'Start workflow on incoming HTTP request',
        phase: 'trigger',
        icon: Webhook,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'trigger_webhook',
        vendor_agnostic: false
    },
    'trigger_schedule': {
        id: 'trigger_schedule',
        name: 'Schedule',
        description: 'Run workflow at fixed intervals',
        phase: 'trigger',
        icon: Clock,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'trigger_schedule',
        vendor_agnostic: false
    },
    'trigger_on_alert': {
        id: 'trigger_on_alert',
        name: 'Alert Triggered',
        description: 'Start workflow when SIEM alert fires',
        phase: 'trigger',
        icon: AlertCircle,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'trigger_alert',
        vendor_agnostic: true // Works with any SIEM
    },

    // ==================== INVESTIGATION ====================
    'analyze_threat': {
        id: 'analyze_threat',
        name: 'Analyze Threat',
        description: 'AI-powered threat analysis and triage',
        phase: 'investigation',
        icon: Bot,
        risk_level: 'low',
        estimated_duration: 30,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'analyze_threat',
        vendor_agnostic: true,
        parameters: {
            required: ['entity'],
            optional: ['depth', 'include_context'],
            schema: {
                entity: { type: 'string', description: 'IP, domain, hash, or user to analyze' },
                depth: { type: 'string', enum: ['quick', 'standard', 'deep'], default: 'standard' },
                include_context: { type: 'boolean', default: true }
            }
        }
    },
    'query_logs': {
        id: 'query_logs',
        name: 'Query Logs',
        description: 'Search SIEM/logs for related activity',
        phase: 'investigation',
        icon: Search,
        risk_level: 'low',
        estimated_duration: 15,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'query_logs',
        vendor_agnostic: true, // Works with Splunk, Elastic, etc.
        parameters: {
            required: ['query'],
            optional: ['timeframe', 'limit'],
            schema: {
                query: { type: 'string', description: 'Search query' },
                timeframe: { type: 'string', default: '1h' },
                limit: { type: 'number', default: 100 }
            }
        }
    },
    'lookup_indicator': {
        id: 'lookup_indicator',
        name: 'Lookup Indicator',
        description: 'Check IP/domain/hash reputation',
        phase: 'investigation',
        icon: FileSearch,
        risk_level: 'low',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'lookup_indicator',
        vendor_agnostic: true, // Uses VirusTotal, etc.
        parameters: {
            required: ['indicator'],
            optional: ['indicator_type'],
            schema: {
                indicator: { type: 'string', description: 'IP, domain, or hash to lookup' },
                indicator_type: { type: 'string', enum: ['ip', 'domain', 'hash', 'auto'], default: 'auto' }
            }
        }
    },
    'get_entity_info': {
        id: 'get_entity_info',
        name: 'Get Entity Details',
        description: 'Retrieve user/host/asset information',
        phase: 'investigation',
        icon: Database,
        risk_level: 'low',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'get_entity_info',
        vendor_agnostic: true
    },
    'extract_iocs': {
        id: 'extract_iocs',
        name: 'Extract IOCs',
        description: 'Pull indicators from logs/artifacts',
        phase: 'investigation',
        icon: Target,
        risk_level: 'low',
        estimated_duration: 10,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'extract_iocs',
        vendor_agnostic: true
    },

    // ==================== CONTAINMENT ====================
    'block_ip': {
        id: 'block_ip',
        name: 'Block IP Address',
        description: 'Block IP at network perimeter',
        phase: 'containment',
        icon: Ban,
        risk_level: 'high',
        estimated_duration: 30,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'block_ip',
        vendor_agnostic: true, // AI picks: Palo Alto, Cisco, Fortinet, etc.
        parameters: {
            required: ['ip'],
            optional: ['duration', 'reason'],
            schema: {
                ip: { type: 'string', description: 'IP address to block' },
                duration: { type: 'string', description: 'Block duration (e.g., "1h", "permanent")', default: 'permanent' },
                reason: { type: 'string', description: 'Reason for blocking' }
            }
        }
    },
    'block_domain': {
        id: 'block_domain',
        name: 'Block Domain',
        description: 'Block domain via DNS/web filtering',
        phase: 'containment',
        icon: Globe,
        risk_level: 'high',
        estimated_duration: 30,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'block_domain',
        vendor_agnostic: true
    },
    'isolate_host': {
        id: 'isolate_host',
        name: 'Isolate Host',
        description: 'Network contain endpoint via EDR',
        phase: 'containment',
        icon: Shield,
        risk_level: 'critical',
        estimated_duration: 10,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'isolate_host',
        vendor_agnostic: true, // AI picks: CrowdStrike, SentinelOne, Defender
        parameters: {
            required: ['hostname_or_ip'],
            optional: ['reason'],
            schema: {
                hostname_or_ip: { type: 'string', description: 'Hostname or IP of host to isolate' },
                reason: { type: 'string', description: 'Reason for isolation' }
            }
        }
    },
    'quarantine_email': {
        id: 'quarantine_email',
        name: 'Quarantine Email',
        description: 'Remove/isolate email messages',
        phase: 'containment',
        icon: Mail,
        risk_level: 'medium',
        estimated_duration: 15,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'quarantine_email',
        vendor_agnostic: true // Works with O365, Proofpoint, etc.
    },

    // ==================== ACCESS CONTROL ====================
    'disable_account': {
        id: 'disable_account',
        name: 'Disable Account',
        description: 'Suspend user access immediately',
        phase: 'access_control',
        icon: UserX,
        risk_level: 'high',
        estimated_duration: 5,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'disable_account',
        vendor_agnostic: true, // Works with Okta, Azure AD, AD
        parameters: {
            required: ['username_or_email'],
            optional: ['reason'],
            schema: {
                username_or_email: { type: 'string', description: 'Username or email to disable' },
                reason: { type: 'string', description: 'Reason for disabling account' }
            }
        }
    },
    'revoke_sessions': {
        id: 'revoke_sessions',
        name: 'Revoke User Sessions',
        description: 'Kill all active user sessions',
        phase: 'access_control',
        icon: Lock,
        risk_level: 'high',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: true,
        intent_type: 'revoke_sessions',
        vendor_agnostic: true
    },
    'reset_password': {
        id: 'reset_password',
        name: 'Reset Password',
        description: 'Force user password change',
        phase: 'access_control',
        icon: Key,
        risk_level: 'medium',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'reset_password',
        vendor_agnostic: true
    },
    'revoke_api_keys': {
        id: 'revoke_api_keys',
        name: 'Revoke API Keys',
        description: 'Deactivate cloud/API credentials',
        phase: 'access_control',
        icon: Cloud,
        risk_level: 'high',
        estimated_duration: 10,
        rollback_supported: false,
        requires_approval: true,
        intent_type: 'revoke_api_keys',
        vendor_agnostic: true // Works with AWS, Azure, GCP
    },

    // ==================== FORENSICS ====================
    'snapshot_system': {
        id: 'snapshot_system',
        name: 'Snapshot System',
        description: 'Create forensic image of VM/disk',
        phase: 'forensics',
        icon: HardDrive,
        risk_level: 'low',
        estimated_duration: 120,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'snapshot_system',
        vendor_agnostic: true
    },
    'collect_artifacts': {
        id: 'collect_artifacts',
        name: 'Collect Artifacts',
        description: 'Gather files, memory, logs for analysis',
        phase: 'forensics',
        icon: FileSearch,
        risk_level: 'low',
        estimated_duration: 60,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'collect_artifacts',
        vendor_agnostic: true
    },
    'get_process_list': {
        id: 'get_process_list',
        name: 'Get Process List',
        description: 'Snapshot running processes',
        phase: 'forensics',
        icon: Terminal,
        risk_level: 'low',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'get_process_list',
        vendor_agnostic: true
    },

    // ==================== REMEDIATION ====================
    'kill_process': {
        id: 'kill_process',
        name: 'Kill Process',
        description: 'Terminate malicious process',
        phase: 'remediation',
        icon: Terminal,
        risk_level: 'high',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: true,
        intent_type: 'kill_process',
        vendor_agnostic: true
    },
    'delete_file': {
        id: 'delete_file',
        name: 'Delete File',
        description: 'Remove malicious file/artifact',
        phase: 'remediation',
        icon: Trash2,
        risk_level: 'high',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: true,
        intent_type: 'delete_file',
        vendor_agnostic: true
    },
    'auto_remediate': {
        id: 'auto_remediate',
        name: 'Auto Remediate',
        description: 'EDR-assisted automated cleanup',
        phase: 'remediation',
        icon: RefreshCw,
        risk_level: 'high',
        estimated_duration: 60,
        rollback_supported: true,
        requires_approval: true,
        intent_type: 'auto_remediate',
        vendor_agnostic: true
    },

    // ==================== COMMUNICATION ====================
    'alert_team': {
        id: 'alert_team',
        name: 'Alert Team',
        description: 'Notify via Slack/Teams/Email',
        phase: 'communication',
        icon: MessageSquare,
        risk_level: 'low',
        estimated_duration: 2,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'alert_team',
        vendor_agnostic: true,
        parameters: {
            required: ['message'],
            optional: ['channel', 'severity'],
            schema: {
                message: { type: 'string', description: 'Alert message' },
                channel: { type: 'string', description: 'Slack/Teams channel' },
                severity: { type: 'string', enum: ['info', 'warning', 'critical'], default: 'warning' }
            }
        }
    },
    'create_ticket': {
        id: 'create_ticket',
        name: 'Create Ticket',
        description: 'Create ServiceNow/Jira ticket',
        phase: 'communication',
        icon: AlertCircle,
        risk_level: 'low',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'create_ticket',
        vendor_agnostic: true
    },
    'escalate_incident': {
        id: 'escalate_incident',
        name: 'Escalate Incident',
        description: 'Page on-call via PagerDuty',
        phase: 'communication',
        icon: AlertCircle,
        risk_level: 'low',
        estimated_duration: 5,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'escalate_incident',
        vendor_agnostic: true
    },

    // ==================== CORE LOGIC ====================
    'if_condition': {
        id: 'if_condition',
        name: 'If Condition',
        description: 'Conditional branching logic',
        phase: 'core',
        icon: Target,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'if_condition',
        vendor_agnostic: false
    },
    'wait': {
        id: 'wait',
        name: 'Wait/Delay',
        description: 'Pause execution for specified duration',
        phase: 'core',
        icon: Pause,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'wait',
        vendor_agnostic: false
    },
    'parallel': {
        id: 'parallel',
        name: 'Run in Parallel',
        description: 'Execute actions simultaneously',
        phase: 'core',
        icon: Play,
        risk_level: 'low',
        estimated_duration: 0,
        rollback_supported: false,
        requires_approval: false,
        intent_type: 'parallel',
        vendor_agnostic: false
    }
};

// Helper to get actions by phase
export function getActionsByPhase(phase: SecurityPhase): IntentAction[] {
    return Object.values(INTENT_BASED_ACTIONS).filter(action => action.phase === phase);
}

// Helper to get high-risk actions
export function getHighRiskActions(): IntentAction[] {
    return Object.values(INTENT_BASED_ACTIONS).filter(
        action => action.risk_level === 'high' || action.risk_level === 'critical'
    );
}

// Helper to check if action requires approval
export function requiresApproval(actionId: string): boolean {
    const action = INTENT_BASED_ACTIONS[actionId];
    return action?.requires_approval || false;
}
