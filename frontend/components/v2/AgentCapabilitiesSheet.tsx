"use client";

import React, { useState, useCallback, useEffect } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Network, Terminal, Database,
  FileSearch, Users, AlertOctagon, Activity,
  Search, Play, Info, Zap, AlertTriangle,
  Loader2, CheckCircle, XCircle, ExternalLink,
  Clock, RotateCcw
} from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { executeAction, ActionExecutionResult, ActionContext } from "@/lib/actionExecutor";
import { toast } from "@/components/ui/toast";

// Types
interface ActionDefinition {
  id: string;
  name: string;
  description: string;
  requiresApproval: boolean;
  impact: string;
}

interface CategoryDefinition {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  actions: ActionDefinition[];
}

interface AgentCapabilitiesSheetProps {
  isOpen: boolean;
  onClose: () => void;
  incidentId: number;
  sourceIp?: string;
  targetHost?: string;
  affectedUserId?: string;
  affectedDomain?: string;
  onActionComplete?: (result: ActionExecutionResult) => void;
  onActionError?: (error: ActionExecutionResult) => void;
}

// Execution state type
interface ExecutionState {
  status: "idle" | "executing" | "completed" | "failed";
  result?: ActionExecutionResult;
  startedAt?: number;
}

// Parameter configuration for actions that need user input
interface ActionParamConfig {
  key: string;
  label: string;
  type: "text" | "number" | "select";
  placeholder?: string;
  defaultValue?: string | number;
  options?: { value: string; label: string }[];
  required?: boolean;
}

// Define parameters for specific actions
const ACTION_PARAMS: Record<string, ActionParamConfig[]> = {
  block_ip_advanced: [
    { key: "duration_seconds", label: "Block Duration", type: "select", defaultValue: "3600", options: [
      { value: "1800", label: "30 minutes" },
      { value: "3600", label: "1 hour" },
      { value: "7200", label: "2 hours" },
      { value: "86400", label: "24 hours" },
      { value: "604800", label: "7 days" },
    ]},
  ],
  isolate_host_advanced: [
    { key: "isolation_level", label: "Isolation Level", type: "select", defaultValue: "full", options: [
      { value: "partial", label: "Partial (Allow management traffic)" },
      { value: "full", label: "Full (Complete isolation)" },
    ]},
    { key: "duration_seconds", label: "Duration", type: "select", defaultValue: "3600", options: [
      { value: "1800", label: "30 minutes" },
      { value: "3600", label: "1 hour" },
      { value: "7200", label: "2 hours" },
      { value: "0", label: "Until manual release" },
    ]},
  ],
  dns_sinkhole: [
    { key: "domains", label: "Domains (comma-separated)", type: "text", placeholder: "malicious.com, bad.net" },
  ],
  process_termination: [
    { key: "process_name", label: "Process Name", type: "text", placeholder: "e.g., malware.exe" },
    { key: "pid", label: "Process ID (optional)", type: "number", placeholder: "e.g., 1234" },
  ],
  scan_endpoint: [
    { key: "scan_type", label: "Scan Type", type: "select", defaultValue: "full", options: [
      { value: "quick", label: "Quick Scan" },
      { value: "full", label: "Full Scan" },
      { value: "custom", label: "Custom Paths" },
    ]},
  ],
  analyze_logs: [
    { key: "time_range", label: "Time Range", type: "select", defaultValue: "24h", options: [
      { value: "1h", label: "Last hour" },
      { value: "6h", label: "Last 6 hours" },
      { value: "24h", label: "Last 24 hours" },
      { value: "7d", label: "Last 7 days" },
    ]},
  ],
  enable_dlp: [
    { key: "policy_level", label: "Policy Level", type: "select", defaultValue: "strict", options: [
      { value: "monitoring", label: "Monitoring Only" },
      { value: "standard", label: "Standard Enforcement" },
      { value: "strict", label: "Strict Enforcement" },
    ]},
  ],
  stakeholder_notification: [
    { key: "notification_level", label: "Notification Level", type: "select", defaultValue: "executive", options: [
      { value: "team", label: "Security Team" },
      { value: "management", label: "Management" },
      { value: "executive", label: "Executive Leadership" },
    ]},
  ],
};

// Comprehensive catalog from codebase analysis
const AGENT_CAPABILITIES: Record<string, CategoryDefinition> = {
  network: {
    label: "Network & Firewall",
    icon: Network,
    color: "blue",
    actions: [
      { id: "block_ip_advanced", name: "Block IP (Advanced)", description: "Block IP with adaptive duration and threat scoring. Prevents all traffic from target.", requiresApproval: false, impact: "Target IP blocked. Active connections dropped." },
      { id: "deploy_firewall_rules", name: "Deploy Firewall Rules", description: "Push custom firewall rules to network perimeter.", requiresApproval: true, impact: "Global firewall policy updated." },
      { id: "dns_sinkhole", name: "DNS Sinkhole", description: "Redirect malicious domains to sinkhole server.", requiresApproval: false, impact: "Domains unresolvable for all users." },
      { id: "traffic_redirection", name: "Traffic Redirection", description: "Redirect suspicious traffic for analysis.", requiresApproval: true, impact: "Traffic latency may increase." },
      { id: "network_segmentation", name: "Network Segmentation", description: "Isolate network segments to contain lateral movement.", requiresApproval: true, impact: "Inter-VLAN traffic blocked." },
      { id: "capture_network_traffic", name: "Capture Traffic (PCAP)", description: "Full packet capture for forensic analysis.", requiresApproval: false, impact: "High storage usage." },
      { id: "deploy_waf_rules", name: "Deploy WAF Rules", description: "Update Web Application Firewall rules.", requiresApproval: true, impact: "WAF config reload." },
    ]
  },
  endpoint: {
    label: "Endpoint & Host",
    icon: Terminal,
    color: "purple",
    actions: [
      { id: "isolate_host_advanced", name: "Isolate Host", description: "Complete network isolation with rollback capability.", requiresApproval: false, impact: "Host offline. Only admin access allowed." },
      { id: "memory_dump_collection", name: "Memory Dump", description: "Capture RAM snapshot for malware analysis.", requiresApproval: false, impact: "System freeze during dump (~30s)." },
      { id: "process_termination", name: "Kill Process", description: "Terminate malicious process by PID or name.", requiresApproval: false, impact: "Process stopped immediately." },
      { id: "registry_hardening", name: "Registry Hardening", description: "Apply security hardening to Windows Registry.", requiresApproval: true, impact: "System restart may be required." },
      { id: "system_recovery", name: "System Recovery", description: "Restore system to clean checkpoint.", requiresApproval: true, impact: "Data since last backup lost." },
      { id: "malware_removal", name: "Malware Removal", description: "Automated malware cleanup and remediation.", requiresApproval: false, impact: "File deletion." },
      { id: "scan_endpoint", name: "Endpoint Scan", description: "Full antivirus/EDR scan of endpoint.", requiresApproval: false, impact: "High CPU usage." },
    ]
  },
  forensics: {
    label: "Investigation & Forensics",
    icon: FileSearch,
    color: "orange",
    actions: [
      { id: "investigate_behavior", name: "Behavior Analysis", description: "Deep dive into attack patterns and TTPs.", requiresApproval: false, impact: "Read-only analysis." },
      { id: "hunt_similar_attacks", name: "Threat Hunting", description: "Proactive search for IoCs across environment.", requiresApproval: false, impact: "Read-only search." },
      { id: "threat_intel_lookup", name: "Threat Intel Lookup", description: "Query external threat intelligence feeds.", requiresApproval: false, impact: "API quota usage." },
      { id: "collect_evidence", name: "Evidence Collection", description: "Gather and preserve forensic artifacts.", requiresApproval: false, impact: "Read-only collection." },
      { id: "analyze_logs", name: "Log Analysis", description: "Correlate and analyze security logs.", requiresApproval: false, impact: "Heavy query load." },
      { id: "attribution_analysis", name: "Attribution Analysis", description: "Identify threat actor using ML and OSINT.", requiresApproval: false, impact: "Read-only analysis." },
    ]
  },
  identity: {
    label: "Identity & Access",
    icon: Users,
    color: "green",
    actions: [
      { id: "reset_passwords", name: "Reset Passwords (Bulk)", description: "Force password reset for compromised accounts.", requiresApproval: true, impact: "Users forced to relogin." },
      { id: "revoke_user_sessions", name: "Revoke Sessions", description: "Terminate all active user sessions.", requiresApproval: false, impact: "Immediate logout." },
      { id: "disable_user_account", name: "Disable Account", description: "Immediately disable user account.", requiresApproval: false, impact: "User lockout." },
      { id: "enforce_mfa", name: "Enforce MFA", description: "Require multi-factor authentication.", requiresApproval: true, impact: "Login flow change." },
      { id: "privileged_access_review", name: "Privilege Review", description: "Audit and restrict privileged access.", requiresApproval: false, impact: "Read-only audit." },
    ]
  },
  data: {
    label: "Data Protection",
    icon: Database,
    color: "cyan",
    actions: [
      { id: "check_database_integrity", name: "Database Integrity Check", description: "Verify database for tampering.", requiresApproval: false, impact: "Database load increase." },
      { id: "backup_critical_data", name: "Emergency Backup", description: "Create immutable backup of critical data.", requiresApproval: false, impact: "High bandwidth usage." },
      { id: "encrypt_sensitive_data", name: "Data Encryption", description: "Apply encryption to sensitive data at rest.", requiresApproval: true, impact: "Data temporarily unavailable." },
      { id: "enable_dlp", name: "Enable DLP", description: "Activate Data Loss Prevention policies.", requiresApproval: true, impact: "Policy enforcement enabled." },
    ]
  },
  communication: {
    label: "Alerting & Notification",
    icon: AlertOctagon,
    color: "red",
    actions: [
      { id: "alert_security_analysts", name: "Alert Analysts", description: "Send urgent notification to SOC team.", requiresApproval: false, impact: "Notifications sent." },
      { id: "create_incident_case", name: "Create Case", description: "Generate incident case in ticketing system.", requiresApproval: false, impact: "Ticket created." },
      { id: "stakeholder_notification", name: "Notify Stakeholders", description: "Alert executive leadership.", requiresApproval: true, impact: "High-priority alert sent." },
    ]
  }
};

export default function AgentCapabilitiesSheet({
  isOpen,
  onClose,
  incidentId,
  sourceIp,
  targetHost,
  affectedUserId,
  affectedDomain,
  onActionComplete,
  onActionError
}: AgentCapabilitiesSheetProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [confirmAction, setConfirmAction] = useState<ActionDefinition | null>(null);
  const [actionParams, setActionParams] = useState<Record<string, string | number>>({});
  const [forceCloseWarning, setForceCloseWarning] = useState(false);

  // Execution state management
  const [executionStates, setExecutionStates] = useState<Record<string, ExecutionState>>({});
  const [activeExecution, setActiveExecution] = useState<{
    action: ActionDefinition;
    state: ExecutionState;
  } | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);

  // Initialize action params with defaults when action is selected
  useEffect(() => {
    if (confirmAction) {
      const paramConfigs = ACTION_PARAMS[confirmAction.id] || [];
      const initialParams: Record<string, string | number> = {};
      paramConfigs.forEach(config => {
        if (config.defaultValue !== undefined) {
          initialParams[config.key] = config.defaultValue;
        }
      });
      setActionParams(initialParams);
    } else {
      setActionParams({});
    }
  }, [confirmAction]);

  // Update elapsed time for executing actions
  useEffect(() => {
    if (activeExecution?.state.status === "executing" && activeExecution.state.startedAt) {
      const interval = setInterval(() => {
        setElapsedTime(Math.round((Date.now() - activeExecution.state.startedAt!) / 1000));
      }, 1000);
      return () => clearInterval(interval);
    } else {
      setElapsedTime(0);
    }
  }, [activeExecution?.state.status, activeExecution?.state.startedAt]);

  // Get execution state for an action
  const getExecutionState = (actionId: string): ExecutionState => {
    return executionStates[actionId] || { status: "idle" };
  };

  // Update execution state
  const updateExecutionState = useCallback((actionId: string, state: Partial<ExecutionState>) => {
    setExecutionStates(prev => ({
      ...prev,
      [actionId]: { ...prev[actionId], ...state } as ExecutionState
    }));
  }, []);

  const filteredCategories: [string, CategoryDefinition][] = Object.entries(AGENT_CAPABILITIES).filter(([key, category]) => {
    if (selectedCategory !== "all" && selectedCategory !== key) return false;

    if (!searchQuery) return true;

    const query = searchQuery.toLowerCase();
    return category.label.toLowerCase().includes(query) ||
           category.actions.some(a =>
             a.name.toLowerCase().includes(query) ||
             a.description.toLowerCase().includes(query)
           );
  });

  const handleActionClick = (action: ActionDefinition) => {
    const state = getExecutionState(action.id);
    // If already executing, don't allow re-click
    if (state.status === "executing") return;
    setConfirmAction(action);
  };

  const handleConfirmExecute = async () => {
    if (!confirmAction || !incidentId) return;

    const actionId = confirmAction.id;
    const action = confirmAction;
    const startTime = Date.now();

    // Update state to executing
    updateExecutionState(actionId, {
      status: "executing",
      startedAt: startTime,
      result: undefined
    });

    // Show execution progress in dialog
    setActiveExecution({
      action,
      state: { status: "executing", startedAt: startTime }
    });

    // Show loading toast
    const loadingToastId = toast.loading(
      `Executing ${action.name}`,
      "Please wait while the action is being processed..."
    );

    // Build context with all available parameters including user-supplied values
    const context: ActionContext = {
      incidentId,
      sourceIp,
      targetHost,
      userId: affectedUserId,
      domain: affectedDomain || (actionParams.domains as string),
      additionalParams: Object.keys(actionParams).length > 0 ? actionParams : undefined,
    };

    // Create a timeout promise (60 seconds for SSH operations)
    const TIMEOUT_MS = 60000;
    const timeoutPromise = new Promise<ActionExecutionResult>((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Action timed out after ${TIMEOUT_MS / 1000}s. The action may have completed on the server - please check the incident status.`));
      }, TIMEOUT_MS);
    });

    try {
      // Execute the action with timeout
      const result = await Promise.race([
        executeAction(actionId, context),
        timeoutPromise
      ]);

      // Calculate actual duration
      const duration = Date.now() - startTime;

      // Dismiss loading toast
      toast.dismiss(loadingToastId);

      // Ensure result has duration
      const resultWithDuration = {
        ...result,
        duration: result.duration || duration,
      };

      // Update state based on result
      const newState: ExecutionState = {
        status: result.success ? "completed" : "failed",
        result: resultWithDuration,
      };

      updateExecutionState(actionId, newState);
      setActiveExecution({
        action,
        state: newState
      });

      // Show result toast
      if (result.success) {
        toast.success(
          `${action.name} Completed`,
          result.message || "Action executed successfully"
        );
        onActionComplete?.(resultWithDuration);
      } else {
        toast.error(
          `${action.name} Failed`,
          result.message || "Action execution failed"
        );
        onActionError?.(resultWithDuration);
      }

      // Auto-close dialog after 3 seconds on success
      if (result.success) {
        setTimeout(() => {
          setActiveExecution(null);
          setConfirmAction(null);
        }, 3000);
      }
    } catch (error) {
      const duration = Date.now() - startTime;
      const isTimeout = String(error).includes('timed out');

      // Dismiss loading toast
      toast.dismiss(loadingToastId);

      const errorResult: ActionExecutionResult = {
        success: false,
        actionId,
        actionName: action.name,
        status: "failed",
        message: isTimeout
          ? "Action timed out - it may have completed on the server. Please refresh to check status."
          : String(error),
        error: String(error),
        timestamp: new Date().toISOString(),
        duration,
      };

      updateExecutionState(actionId, {
        status: "failed",
        result: errorResult
      });

      setActiveExecution({
        action,
        state: { status: "failed", result: errorResult }
      });

      // Show error toast with appropriate message
      toast.error(
        isTimeout ? `${action.name} Timed Out` : `${action.name} Failed`,
        isTimeout
          ? "The action may have completed. Refresh to check status."
          : String(error)
      );

      onActionError?.(errorResult);
    }
  };

  const handleCloseDialog = (force: boolean = false) => {
    if (activeExecution?.state.status === "executing") {
      if (force) {
        // User confirmed force close
        setActiveExecution(null);
        setConfirmAction(null);
        setForceCloseWarning(false);
        toast.warning(
          "Dialog Closed",
          "The action may still be running on the server. Refresh to check status."
        );
      } else {
        // Show force close warning
        setForceCloseWarning(true);
      }
      return;
    }
    setActiveExecution(null);
    setConfirmAction(null);
    setForceCloseWarning(false);
  };

  const handleRetry = () => {
    if (confirmAction) {
      setActiveExecution(null);
      handleConfirmExecute();
    }
  };

  // Get status indicator for action card
  const getActionStatusIndicator = (actionId: string) => {
    const state = getExecutionState(actionId);

    switch (state.status) {
      case "executing":
        return (
          <Badge className="text-[8px] h-4 px-1 bg-blue-500/10 text-blue-500 border-blue-500/30 gap-1">
            <Loader2 className="w-2 h-2 animate-spin" />
            Running
          </Badge>
        );
      case "completed":
        return (
          <Badge className="text-[8px] h-4 px-1 bg-green-500/10 text-green-500 border-green-500/30 gap-1">
            <CheckCircle className="w-2 h-2" />
            Done
          </Badge>
        );
      case "failed":
        return (
          <Badge className="text-[8px] h-4 px-1 bg-red-500/10 text-red-500 border-red-500/30 gap-1">
            <XCircle className="w-2 h-2" />
            Failed
          </Badge>
        );
      default:
        return null;
    }
  };

  // Render execution result in dialog
  const renderExecutionResult = () => {
    if (!activeExecution) return null;

    const { action, state } = activeExecution;

    if (state.status === "executing") {
      const isLongRunning = elapsedTime > 10;
      const isVeryLong = elapsedTime > 30;

      return (
        <div className="py-8 flex flex-col items-center gap-4">
          <div className="relative">
            <div className={cn(
              "w-16 h-16 rounded-full flex items-center justify-center transition-colors",
              isVeryLong ? "bg-yellow-500/10" : "bg-blue-500/10"
            )}>
              <Loader2 className={cn(
                "w-8 h-8 animate-spin",
                isVeryLong ? "text-yellow-500" : "text-blue-500"
              )} />
            </div>
            <div className={cn(
              "absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-background border-2 flex items-center justify-center",
              isVeryLong ? "border-yellow-500" : "border-blue-500"
            )}>
              <Clock className={cn(
                "w-3 h-3",
                isVeryLong ? "text-yellow-500" : "text-blue-500"
              )} />
            </div>
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-foreground">Executing {action.name}</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {isVeryLong
                ? "This is taking longer than expected. The action may still complete..."
                : "Please wait while the action is being processed..."}
            </p>
            {state.startedAt && (
              <p className={cn(
                "text-xs mt-2 font-mono",
                isVeryLong ? "text-yellow-600 dark:text-yellow-400" : "text-muted-foreground"
              )}>
                Elapsed: {elapsedTime}s
              </p>
            )}
            {isLongRunning && (
              <p className="text-xs text-muted-foreground mt-3 max-w-xs">
                <Info className="w-3 h-3 inline mr-1" />
                SSH operations to T-Pot may take 30-60 seconds. You can cancel and refresh to check status.
              </p>
            )}
          </div>
        </div>
      );
    }

    if (state.status === "completed" && state.result) {
      return (
        <div className="py-6 flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center">
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-green-600">Action Completed Successfully</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {state.result.message}
            </p>
            {state.result.duration && (
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                Duration: {(state.result.duration / 1000).toFixed(2)}s
              </p>
            )}
          </div>

          {/* Result Data */}
          {state.result.data && Object.keys(state.result.data).length > 0 && (
            <div className="w-full mt-2">
              <div className="text-xs font-medium text-foreground mb-2">Result Details:</div>
              <div className="bg-muted/50 border rounded-lg p-3 max-h-32 overflow-auto">
                <pre className="text-xs text-muted-foreground font-mono whitespace-pre-wrap">
                  {JSON.stringify(state.result.data, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {state.result.rollbackId && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground bg-blue-500/5 px-3 py-2 rounded-md border border-blue-500/20">
              <RotateCcw className="w-3 h-3 text-blue-500" />
              Rollback available: <code className="font-mono text-blue-600">{state.result.rollbackId}</code>
            </div>
          )}
        </div>
      );
    }

    if (state.status === "failed" && state.result) {
      return (
        <div className="py-6 flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center">
            <XCircle className="w-8 h-8 text-red-500" />
          </div>
          <div className="text-center">
            <h3 className="font-semibold text-red-600">Action Failed</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {state.result.message}
            </p>
          </div>

          {state.result.error && (
            <div className="w-full mt-2">
              <div className="text-xs font-medium text-red-600 mb-2">Error Details:</div>
              <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3 max-h-32 overflow-auto">
                <pre className="text-xs text-red-600 font-mono whitespace-pre-wrap">
                  {state.result.error}
                </pre>
              </div>
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <>
      <Sheet open={isOpen} onOpenChange={onClose}>
        <SheetContent className="w-[900px] sm:max-w-[900px] p-0 flex flex-col bg-background border-l border-border">
          <SheetHeader className="p-6 pb-4 border-b border-border bg-card/50">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Zap className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <SheetTitle className="text-xl font-bold">AI Agent Capabilities</SheetTitle>
                <SheetDescription className="text-xs mt-1 text-muted-foreground">
                  Execute advanced containment, investigation, and remediation actions
                </SheetDescription>
              </div>
              <div className="text-right">
                <Badge variant="outline" className="text-[10px] font-mono">
                  Incident #{incidentId}
                </Badge>
                {sourceIp && (
                  <div className="text-[10px] text-muted-foreground mt-1 font-mono">
                    Target: {sourceIp}
                  </div>
                )}
              </div>
            </div>

            <div className="relative mt-4">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search capabilities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background border-input"
              />
            </div>
          </SheetHeader>

          <Tabs value={selectedCategory} onValueChange={setSelectedCategory} className="flex-1 flex flex-col min-h-0 bg-background">
            <div className="px-6 pt-4 bg-background">
              <TabsList className="w-full grid grid-cols-7 gap-1 h-auto bg-muted/50 p-1">
                <TabsTrigger value="all" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">All</TabsTrigger>
                <TabsTrigger value="network" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Network</TabsTrigger>
                <TabsTrigger value="endpoint" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Endpoint</TabsTrigger>
                <TabsTrigger value="forensics" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Forensics</TabsTrigger>
                <TabsTrigger value="identity" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Identity</TabsTrigger>
                <TabsTrigger value="data" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Data</TabsTrigger>
                <TabsTrigger value="communication" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Alerts</TabsTrigger>
              </TabsList>
            </div>

            <ScrollArea className="flex-1 px-6 py-4 bg-background">
              <div className="space-y-6">
                {filteredCategories.map(([key, category]) => (
                  <div key={key} className="space-y-3">
                    <div className="flex items-center gap-2 sticky top-0 bg-background py-2 border-b border-border z-10">
                      <category.icon className={cn("w-4 h-4", `text-${category.color}-500`)} />
                      <h3 className="font-semibold text-sm text-foreground">{category.label}</h3>
                      <Badge variant="outline" className="ml-auto text-[10px] border-border text-muted-foreground">
                        {category.actions.length} actions
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {category.actions.map((action) => {
                        const execState = getExecutionState(action.id);
                        const isExecuting = execState.status === "executing";

                        return (
                          <div
                            key={action.id}
                            className={cn(
                              "p-3 border rounded-lg bg-card transition-all group cursor-pointer flex flex-col justify-between",
                              isExecuting
                                ? "border-blue-500/50 bg-blue-500/5"
                                : "border-border hover:border-primary/50 hover:bg-accent/50",
                              execState.status === "completed" && "border-green-500/30",
                              execState.status === "failed" && "border-red-500/30"
                            )}
                            onClick={() => handleActionClick(action)}
                          >
                            <div>
                              <div className="flex items-start justify-between mb-2 gap-2">
                                <h4 className={cn(
                                  "font-medium text-sm transition-colors",
                                  isExecuting ? "text-blue-500" : "text-foreground group-hover:text-primary"
                                )}>
                                  {action.name}
                                </h4>
                                <div className="flex items-center gap-1 shrink-0">
                                  {getActionStatusIndicator(action.id)}
                                  {action.requiresApproval && execState.status === "idle" && (
                                    <Badge variant="outline" className="text-[8px] h-4 px-1 bg-yellow-500/10 text-yellow-600 border-yellow-500/30">
                                      Approval
                                    </Badge>
                                  )}
                                </div>
                              </div>
                              <p className="text-xs text-muted-foreground leading-relaxed mb-3">
                                {action.description}
                              </p>
                            </div>
                            <Button
                              size="sm"
                              className={cn(
                                "w-full h-7 text-[10px] gap-1 transition-colors mt-auto",
                                isExecuting
                                  ? "bg-blue-500/10 text-blue-500 cursor-not-allowed"
                                  : "bg-secondary text-secondary-foreground hover:bg-primary hover:text-primary-foreground"
                              )}
                              variant="ghost"
                              disabled={isExecuting}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleActionClick(action);
                              }}
                            >
                              {isExecuting ? (
                                <>
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  Executing...
                                </>
                              ) : (
                                <>
                                  <Play className="w-3 h-3" />
                                  Execute Action
                                </>
                              )}
                            </Button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}

                {filteredCategories.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-20" />
                    <p className="text-sm">No capabilities match your search</p>
                  </div>
                )}
              </div>
            </ScrollArea>

            <div className="border-t border-border p-4 bg-card/50 flex items-center justify-between shrink-0">
              <div className="text-xs text-muted-foreground">
                <Info className="w-3 h-3 inline mr-1" />
                Actions marked with "Approval" require manual confirmation
              </div>
              <Button variant="outline" size="sm" onClick={onClose} className="bg-background hover:bg-accent">
                Close
              </Button>
            </div>
          </Tabs>
        </SheetContent>
      </Sheet>

      {/* Confirmation/Execution Dialog */}
      <Dialog open={!!confirmAction} onOpenChange={(open) => !open && handleCloseDialog(false)}>
        <DialogContent className="bg-background border-border sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              {activeExecution?.state.status === "executing" ? (
                <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
              ) : activeExecution?.state.status === "completed" ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : activeExecution?.state.status === "failed" ? (
                <XCircle className="w-5 h-5 text-red-500" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-warning" />
              )}
              {activeExecution
                ? activeExecution.state.status === "executing"
                  ? "Executing Action"
                  : activeExecution.state.status === "completed"
                    ? "Action Completed"
                    : activeExecution.state.status === "failed"
                      ? "Action Failed"
                      : "Confirm Execution"
                : "Confirm Action Execution"
              }
            </DialogTitle>
            {!activeExecution && (
              <DialogDescription>
                Are you sure you want to execute this action? This may affect system operations.
              </DialogDescription>
            )}
          </DialogHeader>

          {activeExecution ? (
            renderExecutionResult()
          ) : confirmAction && (
            <div className="grid gap-4 py-4">
              <div className="p-4 rounded-md bg-muted/50 border border-border">
                <div className="font-semibold text-sm text-foreground mb-1">{confirmAction.name}</div>
                <div className="text-xs text-muted-foreground mb-3">{confirmAction.description}</div>

                <div className="text-xs font-medium text-foreground mb-1">Expected Impact:</div>
                <div className="text-xs text-muted-foreground bg-background p-2 rounded border border-border">
                  {confirmAction.impact}
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/30 p-2 rounded border">
                <Info className="w-4 h-4" />
                <div>
                  <span className="font-medium">Target:</span> Incident #{incidentId}
                  {sourceIp && <span className="ml-2">| IP: {sourceIp}</span>}
                </div>
              </div>

              {/* Action-specific parameter inputs */}
              {ACTION_PARAMS[confirmAction.id] && ACTION_PARAMS[confirmAction.id].length > 0 && (
                <div className="space-y-3 border border-border rounded-md p-3 bg-background">
                  <div className="text-xs font-medium text-foreground">Action Parameters</div>
                  {ACTION_PARAMS[confirmAction.id].map((param) => (
                    <div key={param.key} className="space-y-1">
                      <Label htmlFor={param.key} className="text-xs text-muted-foreground">
                        {param.label}
                        {param.required && <span className="text-red-500 ml-1">*</span>}
                      </Label>
                      {param.type === "select" ? (
                        <Select
                          value={String(actionParams[param.key] || param.defaultValue || "")}
                          onValueChange={(value) => setActionParams(prev => ({ ...prev, [param.key]: value }))}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue placeholder={`Select ${param.label.toLowerCase()}`} />
                          </SelectTrigger>
                          <SelectContent>
                            {param.options?.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value} className="text-xs">
                                {opt.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id={param.key}
                          type={param.type}
                          placeholder={param.placeholder}
                          className="h-8 text-xs"
                          value={actionParams[param.key] || ""}
                          onChange={(e) => setActionParams(prev => ({
                            ...prev,
                            [param.key]: param.type === "number" ? Number(e.target.value) : e.target.value
                          }))}
                        />
                      )}
                    </div>
                  ))}
                </div>
              )}

              {confirmAction.requiresApproval && (
                <div className="flex items-center gap-2 text-xs text-yellow-600 bg-yellow-500/10 p-2 rounded border border-yellow-500/20">
                  <Info className="w-4 h-4" />
                  This action requires secondary approval which will be requested after initiation.
                </div>
              )}
            </div>
          )}

          <DialogFooter className="flex-col sm:flex-row gap-2">
            {/* Force close warning */}
            {forceCloseWarning && activeExecution?.state.status === "executing" && (
              <div className="w-full mb-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded-md text-xs text-yellow-600 dark:text-yellow-400">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Action may still be running. Force close?</span>
                </div>
                <div className="flex gap-2 mt-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    onClick={() => setForceCloseWarning(false)}
                  >
                    Keep Waiting
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    className="h-7 text-xs"
                    onClick={() => handleCloseDialog(true)}
                  >
                    Force Close
                  </Button>
                </div>
              </div>
            )}

            {activeExecution?.state.status === "failed" ? (
              <>
                <Button
                  variant="outline"
                  onClick={() => handleCloseDialog()}
                  className="bg-background text-foreground border-input hover:bg-accent"
                >
                  Close
                </Button>
                <Button
                  onClick={handleRetry}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Retry
                </Button>
              </>
            ) : activeExecution?.state.status === "completed" ? (
              <Button
                onClick={() => handleCloseDialog()}
                className="bg-green-600 text-white hover:bg-green-700 gap-2"
              >
                <CheckCircle className="w-4 h-4" />
                Done
              </Button>
            ) : activeExecution?.state.status === "executing" ? (
              <div className="flex items-center gap-2 w-full justify-between">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => handleCloseDialog()}
                >
                  <XCircle className="w-3 h-3 mr-1" />
                  Cancel
                </Button>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </div>
              </div>
            ) : (
              <>
                <Button
                  variant="outline"
                  onClick={() => handleCloseDialog()}
                  className="bg-background text-foreground border-input hover:bg-accent hover:text-accent-foreground"
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleConfirmExecute}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 gap-2"
                >
                  <Play className="w-4 h-4" />
                  Execute Action
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
