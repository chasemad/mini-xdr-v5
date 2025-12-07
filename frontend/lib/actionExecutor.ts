/**
 * Action Executor Service
 * Maps AI Agent capabilities to their backend API calls with execution tracking
 */

import {
  socBlockIP,
  socIsolateHost,
  socResetPasswords,
  socCheckDBIntegrity,
  socThreatIntelLookup,
  socDeployWAFRules,
  socCaptureTraffic,
  socHuntSimilarAttacks,
  socAlertAnalysts,
  socCreateCase,
  socBehaviorAnalysis,
  socCollectEvidence,
  socAnalyzeLogs,
  socEndpointScan,
  socEmergencyBackup,
  socRevokeUserSessions,
  socDisableUserAccount,
  socEnforceMFA,
  socEnableDLP,
  socNotifyStakeholders,
  socDnsSinkhole,
  socTrafficRedirection,
  socNetworkSegmentation,
  socMemoryDump,
  socKillProcess,
  socMalwareRemoval,
  socRegistryHardening,
  socSystemRecovery,
  socAttributionAnalysis,
  socPrivilegedAccessReview,
  socEncryptSensitiveData,
  executeSingleResponseAction,
} from "@/app/lib/api";

// Action execution result type
export interface ActionExecutionResult {
  success: boolean;
  actionId: string;
  actionName: string;
  status: "completed" | "failed" | "pending" | "requires_approval";
  message: string;
  data?: Record<string, any>;
  error?: string;
  timestamp: string;
  duration?: number;
  rollbackId?: string;
}

// Action execution context
export interface ActionContext {
  incidentId: number;
  sourceIp?: string;
  targetHost?: string;
  userId?: string;
  domain?: string;
  additionalParams?: Record<string, any>;
}

// Action definition type
export interface ActionDefinition {
  id: string;
  name: string;
  description: string;
  category: string;
  requiresApproval: boolean;
  impact: string;
  executor: (context: ActionContext) => Promise<ActionExecutionResult>;
}

// Helper to wrap API responses into ActionExecutionResult
const wrapResult = (
  actionId: string,
  actionName: string,
  startTime: number,
  response: any,
  error?: any
): ActionExecutionResult => {
  const duration = Date.now() - startTime;

  if (error) {
    return {
      success: false,
      actionId,
      actionName,
      status: "failed",
      message: error?.message || "Action execution failed",
      error: String(error),
      timestamp: new Date().toISOString(),
      duration,
    };
  }

  return {
    success: true,
    actionId,
    actionName,
    status: "completed",
    message: response?.message || response?.detail || `${actionName} completed successfully`,
    data: response,
    timestamp: new Date().toISOString(),
    duration,
    rollbackId: response?.rollback_id,
  };
};

// Generic executor using the response engine
const executeViaResponseEngine = async (
  actionType: string,
  actionName: string,
  context: ActionContext,
  params: Record<string, any> = {}
): Promise<ActionExecutionResult> => {
  const startTime = Date.now();

  try {
    const response = await executeSingleResponseAction({
      action_type: actionType,
      incident_id: context.incidentId,
      parameters: {
        ...params,
        source_ip: context.sourceIp,
        target_host: context.targetHost,
        user_id: context.userId,
        ...context.additionalParams,
      },
    });

    return wrapResult(actionType, actionName, startTime, response);
  } catch (error) {
    return wrapResult(actionType, actionName, startTime, null, error);
  }
};

/**
 * Action Executors Map
 * Maps capability IDs to their execution functions
 */
export const ACTION_EXECUTORS: Record<string, ActionDefinition> = {
  // ==================== NETWORK & FIREWALL ====================
  block_ip_advanced: {
    id: "block_ip_advanced",
    name: "Block IP (Advanced)",
    description: "Block IP with adaptive duration and threat scoring",
    category: "network",
    requiresApproval: false,
    impact: "Target IP blocked. Active connections dropped.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const duration = context.additionalParams?.duration_seconds || 3600;
        const response = await socBlockIP(context.incidentId, duration);
        return wrapResult("block_ip_advanced", "Block IP (Advanced)", startTime, response);
      } catch (error) {
        return wrapResult("block_ip_advanced", "Block IP (Advanced)", startTime, null, error);
      }
    },
  },

  deploy_firewall_rules: {
    id: "deploy_firewall_rules",
    name: "Deploy Firewall Rules",
    description: "Push custom firewall rules to network perimeter",
    category: "network",
    requiresApproval: true,
    impact: "Global firewall policy updated.",
    executor: async (context) => {
      return executeViaResponseEngine("deploy_firewall_rules", "Deploy Firewall Rules", context, {
        rule_set: context.additionalParams?.rule_set || "default_block",
        scope: context.additionalParams?.scope || "perimeter",
        priority: "high",
      });
    },
  },

  dns_sinkhole: {
    id: "dns_sinkhole",
    name: "DNS Sinkhole",
    description: "Redirect malicious domains to sinkhole server",
    category: "network",
    requiresApproval: false,
    impact: "Domains unresolvable for all users.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const domains = context.additionalParams?.domains
          ? (context.additionalParams.domains as string).split(',').map(d => d.trim())
          : context.domain ? [context.domain] : undefined;
        const response = await socDnsSinkhole(context.incidentId, domains);
        return wrapResult("dns_sinkhole", "DNS Sinkhole", startTime, response);
      } catch (error) {
        return wrapResult("dns_sinkhole", "DNS Sinkhole", startTime, null, error);
      }
    },
  },

  traffic_redirection: {
    id: "traffic_redirection",
    name: "Traffic Redirection",
    description: "Redirect suspicious traffic for analysis",
    category: "network",
    requiresApproval: true,
    impact: "Traffic latency may increase.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const destination = context.additionalParams?.destination as string || "honeypot";
        const response = await socTrafficRedirection(context.incidentId, destination);
        return wrapResult("traffic_redirection", "Traffic Redirection", startTime, response);
      } catch (error) {
        return wrapResult("traffic_redirection", "Traffic Redirection", startTime, null, error);
      }
    },
  },

  network_segmentation: {
    id: "network_segmentation",
    name: "Network Segmentation",
    description: "Isolate network segments to contain lateral movement",
    category: "network",
    requiresApproval: true,
    impact: "Inter-VLAN traffic blocked.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const segmentType = context.additionalParams?.segment_type as string || "vlan";
        const response = await socNetworkSegmentation(context.incidentId, segmentType);
        return wrapResult("network_segmentation", "Network Segmentation", startTime, response);
      } catch (error) {
        return wrapResult("network_segmentation", "Network Segmentation", startTime, null, error);
      }
    },
  },

  capture_network_traffic: {
    id: "capture_network_traffic",
    name: "Capture Traffic (PCAP)",
    description: "Full packet capture for forensic analysis",
    category: "network",
    requiresApproval: false,
    impact: "High storage usage.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socCaptureTraffic(context.incidentId);
        return wrapResult("capture_network_traffic", "Capture Traffic (PCAP)", startTime, response);
      } catch (error) {
        return wrapResult("capture_network_traffic", "Capture Traffic (PCAP)", startTime, null, error);
      }
    },
  },

  deploy_waf_rules: {
    id: "deploy_waf_rules",
    name: "Deploy WAF Rules",
    description: "Update Web Application Firewall rules",
    category: "network",
    requiresApproval: true,
    impact: "WAF config reload.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socDeployWAFRules(context.incidentId);
        return wrapResult("deploy_waf_rules", "Deploy WAF Rules", startTime, response);
      } catch (error) {
        return wrapResult("deploy_waf_rules", "Deploy WAF Rules", startTime, null, error);
      }
    },
  },

  // ==================== ENDPOINT & HOST ====================
  isolate_host_advanced: {
    id: "isolate_host_advanced",
    name: "Isolate Host",
    description: "Complete network isolation with rollback capability",
    category: "endpoint",
    requiresApproval: false,
    impact: "Host offline. Only admin access allowed.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const isolationLevel = context.additionalParams?.isolation_level || "full";
        const duration = context.additionalParams?.duration_seconds;
        const response = await socIsolateHost(context.incidentId, isolationLevel, duration);
        return wrapResult("isolate_host_advanced", "Isolate Host", startTime, response);
      } catch (error) {
        return wrapResult("isolate_host_advanced", "Isolate Host", startTime, null, error);
      }
    },
  },

  memory_dump_collection: {
    id: "memory_dump_collection",
    name: "Memory Dump",
    description: "Capture RAM snapshot for malware analysis",
    category: "endpoint",
    requiresApproval: false,
    impact: "System freeze during dump (~30s).",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socMemoryDump(context.incidentId, context.targetHost);
        return wrapResult("memory_dump_collection", "Memory Dump", startTime, response);
      } catch (error) {
        return wrapResult("memory_dump_collection", "Memory Dump", startTime, null, error);
      }
    },
  },

  process_termination: {
    id: "process_termination",
    name: "Kill Process",
    description: "Terminate malicious process by PID or name",
    category: "endpoint",
    requiresApproval: false,
    impact: "Process stopped immediately.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const processCriteria = {
          process_name: context.additionalParams?.process_name,
          pid: context.additionalParams?.pid,
        };
        const response = await socKillProcess(context.incidentId, processCriteria);
        return wrapResult("process_termination", "Kill Process", startTime, response);
      } catch (error) {
        return wrapResult("process_termination", "Kill Process", startTime, null, error);
      }
    },
  },

  registry_hardening: {
    id: "registry_hardening",
    name: "Registry Hardening",
    description: "Apply security hardening to Windows Registry",
    category: "endpoint",
    requiresApproval: true,
    impact: "System restart may be required.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const hardeningProfile = context.additionalParams?.hardening_profile as string || "security_baseline";
        const response = await socRegistryHardening(context.incidentId, hardeningProfile, true);
        return wrapResult("registry_hardening", "Registry Hardening", startTime, response);
      } catch (error) {
        return wrapResult("registry_hardening", "Registry Hardening", startTime, null, error);
      }
    },
  },

  system_recovery: {
    id: "system_recovery",
    name: "System Recovery",
    description: "Restore system to clean checkpoint",
    category: "endpoint",
    requiresApproval: true,
    impact: "Data since last backup lost.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const recoveryPoint = context.additionalParams?.recovery_point as string || "latest_clean";
        const response = await socSystemRecovery(context.incidentId, recoveryPoint);
        return wrapResult("system_recovery", "System Recovery", startTime, response);
      } catch (error) {
        return wrapResult("system_recovery", "System Recovery", startTime, null, error);
      }
    },
  },

  malware_removal: {
    id: "malware_removal",
    name: "Malware Removal",
    description: "Automated malware cleanup and remediation",
    category: "endpoint",
    requiresApproval: false,
    impact: "File deletion.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socMalwareRemoval(context.incidentId, context.targetHost);
        return wrapResult("malware_removal", "Malware Removal", startTime, response);
      } catch (error) {
        return wrapResult("malware_removal", "Malware Removal", startTime, null, error);
      }
    },
  },

  scan_endpoint: {
    id: "scan_endpoint",
    name: "Endpoint Scan",
    description: "Full antivirus/EDR scan of endpoint",
    category: "endpoint",
    requiresApproval: false,
    impact: "High CPU usage.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const scanType = context.additionalParams?.scan_type || "full";
        const response = await socEndpointScan(context.incidentId, context.targetHost, scanType);
        return wrapResult("scan_endpoint", "Endpoint Scan", startTime, response);
      } catch (error) {
        return wrapResult("scan_endpoint", "Endpoint Scan", startTime, null, error);
      }
    },
  },

  // ==================== INVESTIGATION & FORENSICS ====================
  investigate_behavior: {
    id: "investigate_behavior",
    name: "Behavior Analysis",
    description: "Deep dive into attack patterns and TTPs",
    category: "forensics",
    requiresApproval: false,
    impact: "Read-only analysis.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socBehaviorAnalysis(context.incidentId);
        return wrapResult("investigate_behavior", "Behavior Analysis", startTime, response);
      } catch (error) {
        return wrapResult("investigate_behavior", "Behavior Analysis", startTime, null, error);
      }
    },
  },

  hunt_similar_attacks: {
    id: "hunt_similar_attacks",
    name: "Threat Hunting",
    description: "Proactive search for IoCs across environment",
    category: "forensics",
    requiresApproval: false,
    impact: "Read-only search.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socHuntSimilarAttacks(context.incidentId);
        return wrapResult("hunt_similar_attacks", "Threat Hunting", startTime, response);
      } catch (error) {
        return wrapResult("hunt_similar_attacks", "Threat Hunting", startTime, null, error);
      }
    },
  },

  threat_intel_lookup: {
    id: "threat_intel_lookup",
    name: "Threat Intel Lookup",
    description: "Query external threat intelligence feeds",
    category: "forensics",
    requiresApproval: false,
    impact: "API quota usage.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socThreatIntelLookup(context.incidentId);
        return wrapResult("threat_intel_lookup", "Threat Intel Lookup", startTime, response);
      } catch (error) {
        return wrapResult("threat_intel_lookup", "Threat Intel Lookup", startTime, null, error);
      }
    },
  },

  collect_evidence: {
    id: "collect_evidence",
    name: "Evidence Collection",
    description: "Gather and preserve forensic artifacts",
    category: "forensics",
    requiresApproval: false,
    impact: "Read-only collection.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const artifactTypes = context.additionalParams?.artifact_types || ["logs", "memory", "registry", "network"];
        const response = await socCollectEvidence(context.incidentId, artifactTypes);
        return wrapResult("collect_evidence", "Evidence Collection", startTime, response);
      } catch (error) {
        return wrapResult("collect_evidence", "Evidence Collection", startTime, null, error);
      }
    },
  },

  analyze_logs: {
    id: "analyze_logs",
    name: "Log Analysis",
    description: "Correlate and analyze security logs",
    category: "forensics",
    requiresApproval: false,
    impact: "Heavy query load.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const timeRange = context.additionalParams?.time_range || "24h";
        const response = await socAnalyzeLogs(context.incidentId, timeRange);
        return wrapResult("analyze_logs", "Log Analysis", startTime, response);
      } catch (error) {
        return wrapResult("analyze_logs", "Log Analysis", startTime, null, error);
      }
    },
  },

  attribution_analysis: {
    id: "attribution_analysis",
    name: "Attribution Analysis",
    description: "Identify threat actor using ML and OSINT",
    category: "forensics",
    requiresApproval: false,
    impact: "Read-only analysis.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socAttributionAnalysis(context.incidentId);
        return wrapResult("attribution_analysis", "Attribution Analysis", startTime, response);
      } catch (error) {
        return wrapResult("attribution_analysis", "Attribution Analysis", startTime, null, error);
      }
    },
  },

  // ==================== IDENTITY & ACCESS ====================
  reset_passwords: {
    id: "reset_passwords",
    name: "Reset Passwords (Bulk)",
    description: "Force password reset for compromised accounts",
    category: "identity",
    requiresApproval: true,
    impact: "Users forced to relogin.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socResetPasswords(context.incidentId);
        return wrapResult("reset_passwords", "Reset Passwords (Bulk)", startTime, response);
      } catch (error) {
        return wrapResult("reset_passwords", "Reset Passwords (Bulk)", startTime, null, error);
      }
    },
  },

  revoke_user_sessions: {
    id: "revoke_user_sessions",
    name: "Revoke Sessions",
    description: "Terminate all active user sessions",
    category: "identity",
    requiresApproval: false,
    impact: "Immediate logout.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socRevokeUserSessions(context.incidentId, context.userId);
        return wrapResult("revoke_user_sessions", "Revoke Sessions", startTime, response);
      } catch (error) {
        return wrapResult("revoke_user_sessions", "Revoke Sessions", startTime, null, error);
      }
    },
  },

  disable_user_account: {
    id: "disable_user_account",
    name: "Disable Account",
    description: "Immediately disable user account",
    category: "identity",
    requiresApproval: false,
    impact: "User lockout.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socDisableUserAccount(context.incidentId, context.userId);
        return wrapResult("disable_user_account", "Disable Account", startTime, response);
      } catch (error) {
        return wrapResult("disable_user_account", "Disable Account", startTime, null, error);
      }
    },
  },

  enforce_mfa: {
    id: "enforce_mfa",
    name: "Enforce MFA",
    description: "Require multi-factor authentication",
    category: "identity",
    requiresApproval: true,
    impact: "Login flow change.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socEnforceMFA(context.incidentId, context.userId);
        return wrapResult("enforce_mfa", "Enforce MFA", startTime, response);
      } catch (error) {
        return wrapResult("enforce_mfa", "Enforce MFA", startTime, null, error);
      }
    },
  },

  privileged_access_review: {
    id: "privileged_access_review",
    name: "Privilege Review",
    description: "Audit and restrict privileged access",
    category: "identity",
    requiresApproval: false,
    impact: "Read-only audit.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const scope = context.additionalParams?.scope as string || "all_privileged";
        const response = await socPrivilegedAccessReview(context.incidentId, scope, true);
        return wrapResult("privileged_access_review", "Privilege Review", startTime, response);
      } catch (error) {
        return wrapResult("privileged_access_review", "Privilege Review", startTime, null, error);
      }
    },
  },

  // ==================== DATA PROTECTION ====================
  check_database_integrity: {
    id: "check_database_integrity",
    name: "Database Integrity Check",
    description: "Verify database for tampering",
    category: "data",
    requiresApproval: false,
    impact: "Database load increase.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socCheckDBIntegrity(context.incidentId);
        return wrapResult("check_database_integrity", "Database Integrity Check", startTime, response);
      } catch (error) {
        return wrapResult("check_database_integrity", "Database Integrity Check", startTime, null, error);
      }
    },
  },

  backup_critical_data: {
    id: "backup_critical_data",
    name: "Emergency Backup",
    description: "Create immutable backup of critical data",
    category: "data",
    requiresApproval: false,
    impact: "High bandwidth usage.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socEmergencyBackup(context.incidentId);
        return wrapResult("backup_critical_data", "Emergency Backup", startTime, response);
      } catch (error) {
        return wrapResult("backup_critical_data", "Emergency Backup", startTime, null, error);
      }
    },
  },

  encrypt_sensitive_data: {
    id: "encrypt_sensitive_data",
    name: "Data Encryption",
    description: "Apply encryption to sensitive data at rest",
    category: "data",
    requiresApproval: true,
    impact: "Data temporarily unavailable.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const algorithm = context.additionalParams?.encryption_algorithm as string || "AES-256";
        const keyManagement = context.additionalParams?.key_management as string || "hsm";
        const response = await socEncryptSensitiveData(context.incidentId, algorithm, keyManagement);
        return wrapResult("encrypt_sensitive_data", "Data Encryption", startTime, response);
      } catch (error) {
        return wrapResult("encrypt_sensitive_data", "Data Encryption", startTime, null, error);
      }
    },
  },

  enable_dlp: {
    id: "enable_dlp",
    name: "Enable DLP",
    description: "Activate Data Loss Prevention policies",
    category: "data",
    requiresApproval: true,
    impact: "Policy enforcement enabled.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const policyLevel = context.additionalParams?.policy_level || "strict";
        const response = await socEnableDLP(context.incidentId, policyLevel);
        return wrapResult("enable_dlp", "Enable DLP", startTime, response);
      } catch (error) {
        return wrapResult("enable_dlp", "Enable DLP", startTime, null, error);
      }
    },
  },

  // ==================== ALERTING & NOTIFICATION ====================
  alert_security_analysts: {
    id: "alert_security_analysts",
    name: "Alert Analysts",
    description: "Send urgent notification to SOC team",
    category: "communication",
    requiresApproval: false,
    impact: "Notifications sent.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socAlertAnalysts(context.incidentId);
        return wrapResult("alert_security_analysts", "Alert Analysts", startTime, response);
      } catch (error) {
        return wrapResult("alert_security_analysts", "Alert Analysts", startTime, null, error);
      }
    },
  },

  create_incident_case: {
    id: "create_incident_case",
    name: "Create Case",
    description: "Generate incident case in ticketing system",
    category: "communication",
    requiresApproval: false,
    impact: "Ticket created.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const response = await socCreateCase(context.incidentId);
        return wrapResult("create_incident_case", "Create Case", startTime, response);
      } catch (error) {
        return wrapResult("create_incident_case", "Create Case", startTime, null, error);
      }
    },
  },

  stakeholder_notification: {
    id: "stakeholder_notification",
    name: "Notify Stakeholders",
    description: "Alert executive leadership",
    category: "communication",
    requiresApproval: true,
    impact: "High-priority alert sent.",
    executor: async (context) => {
      const startTime = Date.now();
      try {
        const notificationLevel = context.additionalParams?.notification_level || "executive";
        const response = await socNotifyStakeholders(context.incidentId, notificationLevel);
        return wrapResult("stakeholder_notification", "Notify Stakeholders", startTime, response);
      } catch (error) {
        return wrapResult("stakeholder_notification", "Notify Stakeholders", startTime, null, error);
      }
    },
  },
};

/**
 * Execute an action by its ID
 */
export async function executeAction(
  actionId: string,
  context: ActionContext
): Promise<ActionExecutionResult> {
  const actionDef = ACTION_EXECUTORS[actionId];

  if (!actionDef) {
    return {
      success: false,
      actionId,
      actionName: actionId,
      status: "failed",
      message: `Unknown action: ${actionId}`,
      error: "Action not found in executor registry",
      timestamp: new Date().toISOString(),
    };
  }

  console.log(`[ActionExecutor] Executing ${actionDef.name} for incident ${context.incidentId}`);

  try {
    const result = await actionDef.executor(context);
    console.log(`[ActionExecutor] ${actionDef.name} completed:`, result);
    return result;
  } catch (error) {
    console.error(`[ActionExecutor] ${actionDef.name} failed:`, error);
    return {
      success: false,
      actionId,
      actionName: actionDef.name,
      status: "failed",
      message: `Action execution failed: ${error}`,
      error: String(error),
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Get action definition by ID
 */
export function getActionDefinition(actionId: string): ActionDefinition | undefined {
  return ACTION_EXECUTORS[actionId];
}

/**
 * Get all action definitions
 */
export function getAllActionDefinitions(): ActionDefinition[] {
  return Object.values(ACTION_EXECUTORS);
}

/**
 * Check if an action requires approval
 */
export function actionRequiresApproval(actionId: string): boolean {
  return ACTION_EXECUTORS[actionId]?.requiresApproval ?? false;
}
