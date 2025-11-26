/**
 * Action Formatters and Utilities
 * Helper functions for formatting and displaying action data
 */

import type { LucideIcon } from 'lucide-react';
import { Shield, Lock, Key, Ban, Network, AlertCircle, FileText, Eye, Users } from 'lucide-react';

export interface ActionIconConfig {
  icon: LucideIcon;
  tone: 'primary' | 'info' | 'success' | 'warning' | 'danger' | 'neutral';
}

export const ACTION_NAME_MAP: Record<string, string> = {
  "block_ip": "Block IP Address",
  "unblock_ip": "Unblock IP Address",
  "isolate_host": "Isolate Host",
  "unisolate_host": "Remove Host Isolation",
  "reset_passwords": "Reset User Passwords",
  "check_db_integrity": "Check Database Integrity",
  "threat_intel_lookup": "Threat Intelligence Lookup",
  "deploy_waf_rules": "Deploy WAF Rules",
  "capture_traffic": "Capture Network Traffic",
  "hunt_similar_attacks": "Hunt Similar Attacks",
  "alert_analysts": "Alert SOC Analysts",
  "create_case": "Create Incident Case",
  "disable_user": "Disable User Account",
  "enable_user": "Enable User Account",
  "force_password_reset": "Force Password Reset",
  "add_to_security_group": "Add to Security Group",
  "remove_from_security_group": "Remove from Security Group",
  "quarantine_file": "Quarantine Malicious File",
  "restore_file": "Restore Quarantined File",
  "kill_process": "Terminate Process",
  "block_domain": "Block Malicious Domain",
  "isolate_network_segment": "Isolate Network Segment",
  "block_sensitive_file_transfer": "Block File Transfer",
  "quarantine_sensitive_file": "Quarantine Sensitive File"
};

export const getActionIcon = (actionKey: string, agentType?: string): ActionIconConfig => {
  // Agent-specific icons
  const agentIcons: Record<string, ActionIconConfig> = {
    iam: {
      icon: Users,
      tone: 'primary',
    },
    edr: {
      icon: Shield,
      tone: 'info',
    },
    dlp: {
      icon: Lock,
      tone: 'success',
    }
  };

  if (agentType && agentIcons[agentType]) {
    return agentIcons[agentType];
  }

  // Action-specific icons
  const actionIcons: Record<string, ActionIconConfig> = {
    block_ip: {
      icon: Ban,
      tone: 'danger',
    },
    isolate_host: {
      icon: Network,
      tone: 'warning',
    },
    disable_user: {
      icon: Users,
      tone: 'primary',
    },
    quarantine_file: {
      icon: FileText,
      tone: 'info',
    },
    reset_passwords: {
      icon: Key,
      tone: 'warning',
    },
    threat_intel_lookup: {
      icon: Eye,
      tone: 'info',
    }
  };

  return actionIcons[actionKey] || {
    icon: AlertCircle,
    tone: 'neutral',
  };
};

export const getStatusColor = (status: string): string => {
  const normalized = status.toLowerCase();
  if (['completed', 'success', 'done'].includes(normalized)) return 'success';
  if (['failed', 'error'].includes(normalized)) return 'danger';
  if (['pending', 'running', 'in_progress'].includes(normalized)) return 'warning';
  return 'neutral';
};

export const getStatusIcon = (status: string) => {
  const normalized = status.toLowerCase();
  if (['completed', 'success', 'done'].includes(normalized)) return '✅';
  if (['failed', 'error'].includes(normalized)) return '❌';
  if (['pending', 'running', 'in_progress'].includes(normalized)) return '⏳';
  if (['rolled_back', 'rollback'].includes(normalized)) return '🔄';
  return '⚪';
};

export const formatTimeAgo = (dateString: string): string => {
  const now = new Date();
  const date = new Date(dateString);
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 10) return 'just now';
  if (diffSecs < 60) return `${diffSecs}s ago`;
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
};

export const formatAbsoluteTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true
  });
};

export const formatDuration = (startTime: string, endTime?: string): string => {
  const start = new Date(startTime);
  const end = endTime ? new Date(endTime) : new Date();
  const diffMs = end.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);

  if (diffSecs < 1) return '< 1s';
  if (diffSecs < 60) return `${diffSecs}s`;
  const diffMins = Math.floor(diffSecs / 60);
  if (diffMins < 60) return `${diffMins}m ${diffSecs % 60}s`;
  const diffHours = Math.floor(diffMins / 60);
  return `${diffHours}h ${diffMins % 60}m`;
};

export const getActionDisplayName = (actionKey: string): string => {
  return ACTION_NAME_MAP[actionKey] || toTitleCase(actionKey);
};

export const toTitleCase = (str: string): string => {
  return str
    .replace(/_/g, ' ')
    .replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
};

export const truncateText = (text: string, maxLength: number = 50): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

export const formatJSON = (obj: unknown): string => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
};

export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy to clipboard:', err);
    return false;
  }
};

export interface ActionRecord {
  status?: string;
  source?: string;
}

export interface ActionSummary {
  totalActions: number;
  manualActions: number;
  workflowActions: number;
  agentActions: number;
  successCount: number;
  failureCount: number;
  pendingCount: number;
  successRate: number;
}

export const calculateActionSummary = (actions: ActionRecord[]): ActionSummary => {
  const manual = actions.filter(a => a.source === 'manual');
  const workflow = actions.filter(a => a.source === 'workflow');
  const agent = actions.filter(a => a.source === 'agent');

  const success = actions.filter(a =>
    ['completed', 'success', 'done'].includes(a.status?.toLowerCase())
  );
  const failure = actions.filter(a =>
    ['failed', 'error'].includes(a.status?.toLowerCase())
  );
  const pending = actions.filter(a =>
    ['pending', 'running', 'in_progress'].includes(a.status?.toLowerCase())
  );

  const successRate = actions.length > 0
    ? (success.length / actions.length) * 100
    : 0;

  return {
    totalActions: actions.length,
    manualActions: manual.length,
    workflowActions: workflow.length,
    agentActions: agent.length,
    successCount: success.length,
    failureCount: failure.length,
    pendingCount: pending.length,
    successRate: Math.round(successRate)
  };
};
