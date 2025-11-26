"use client";

import React, { use, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Clock, Shield, AlertTriangle, User, CheckCircle, MoreVertical, Activity, Share2, Search, MessageSquare, Send, BrainCircuit, ChevronDown, ChevronUp, Bot, History, Zap, Terminal, Menu, Play } from "lucide-react";
import { useIncidentRealtime } from "@/app/hooks/useIncidentRealtime";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

import ThreatScoreCard from "@/components/v2/ThreatScoreCard";
import InteractiveTimeline from "@/components/v2/InteractiveTimeline";
import ActionCard from "@/components/v2/ActionCard";
import ActionHistorySheet from "@/components/v2/ActionHistorySheet";
import AgentCapabilitiesSheet from "@/components/v2/AgentCapabilitiesSheet";
import EntityActionHistory from "@/components/v2/EntityActionHistory";

import { agentApi, type IncidentCoordination } from "@/lib/agent-api";
import DeepAnalysisSheet from "@/components/v2/DeepAnalysisSheet";
import AIAnalysisCard from "@/components/v2/AIAnalysisCard";
import ResponseSection from "@/components/v2/ResponseSection";
import InvestigationWorkspace from "@/components/v2/InvestigationWorkspace";
import EventsHistorySheet from "@/components/v2/EventsHistorySheet";
import { getBlockStatus, socBlockIP, socUnblockIP } from "@/app/lib/api";

export default function IncidentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const incidentId = parseInt(id);
  const router = useRouter();
  const [isAgentActionsOpen, setIsAgentActionsOpen] = useState(true);
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isEventsOpen, setIsEventsOpen] = useState(false);
  const [isCapabilitiesOpen, setIsCapabilitiesOpen] = useState(false);
  const [isDeepAnalysisOpen, setIsDeepAnalysisOpen] = useState(false);
  const [isEntityActionsOpen, setIsEntityActionsOpen] = useState(false);
  const [selectedAgentAction, setSelectedAgentAction] = useState<any>(null);
  const [blockStatus, setBlockStatus] = useState<{ is_blocked: boolean; ip?: string; last_checked?: string } | null>(null);
  const [isCheckingBlock, setIsCheckingBlock] = useState(false);
  const [blockActionLoading, setBlockActionLoading] = useState<"block" | "unblock" | null>(null);

  // Ref to prevent re-renders during background polling
  const isPollingRef = useRef(false);
  const lastBlockStatusRef = useRef<boolean | null>(null);

  const [coordination, setCoordination] = useState<IncidentCoordination | null>(null);
  const [coordinationLoading, setCoordinationLoading] = useState(true);

  // Enhanced AI Analysis state
  const [aiAnalysis, setAiAnalysis] = useState<{
    analysis: any;
    triage_note: any;
    event_count: number;
    ai_analysis_timestamp: string | null;
  } | null>(null);
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false);
  const lastEventCountRef = useRef<number>(0);

  // Fetch AI Analysis
  const fetchAiAnalysis = React.useCallback(async (forceRefresh = false) => {
    if (!incidentId) return;
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
    if (!token) return;

    try {
      setAiAnalysisLoading(true);
      const endpoint = forceRefresh
        ? `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/api/incidents/${incidentId}/refresh-analysis`
        : `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/api/incidents/${incidentId}/ai-analysis`;

      const response = await fetch(endpoint, {
        method: forceRefresh ? "POST" : "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAiAnalysis(data);
        lastEventCountRef.current = data.event_count || 0;
      }
    } catch (error) {
      console.error("Failed to fetch AI analysis:", error);
    } finally {
      setAiAnalysisLoading(false);
    }
  }, [incidentId]);

  React.useEffect(() => {
    if (incidentId) {
      const fetchCoordination = async () => {
        try {
          const data = await agentApi.getIncidentCoordination(incidentId);
          setCoordination(data);
        } catch (e) {
          console.error("Failed to fetch agent coordination:", e);
        } finally {
          setCoordinationLoading(false);
        }
      };

      fetchCoordination();

      // Also fetch AI analysis
      fetchAiAnalysis();
    }
  }, [incidentId, fetchAiAnalysis]);

  // Refresh block status - optimized to reduce re-renders
  const refreshBlockStatus = React.useCallback(async (showLoading = true) => {
    if (!incidentId) return;
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("access_token");
      if (!token) return;
    }

    // Avoid concurrent requests
    if (isPollingRef.current) return;
    isPollingRef.current = true;

    try {
      // Only show loading indicator for manual refreshes, not background polling
      if (showLoading) {
        setIsCheckingBlock(true);
      }

      const status = await getBlockStatus(incidentId);

      if (status) {
        // Only update state if the block status actually changed
        if (lastBlockStatusRef.current !== status.is_blocked) {
          lastBlockStatusRef.current = status.is_blocked;
          setBlockStatus(status);
        }
      }
    } catch (error) {
      // Silently fail for background polling
      if (showLoading) {
        console.error("Failed to fetch block status:", error);
      }
    } finally {
      if (showLoading) {
        setIsCheckingBlock(false);
      }
      isPollingRef.current = false;
    }
  }, [incidentId]);

  const handleActionUpdate = React.useCallback(() => {
    // Refresh block status immediately when actions occur (with loading indicator)
    refreshBlockStatus(true);
  }, [refreshBlockStatus]);

  const {
    incident,
    loading,
    refreshIncident
  } = useIncidentRealtime({
    incidentId,
    autoRefresh: false,
    refreshInterval: 10000,
    onNewAction: handleActionUpdate,
    onActionComplete: handleActionUpdate,
    onStatusChange: handleActionUpdate
  });

  // Initial block status check
  React.useEffect(() => {
    refreshBlockStatus(true); // Show loading on initial fetch
  }, [refreshBlockStatus]);

  // Background polling for block status - less frequent to avoid UI disruption
  React.useEffect(() => {
    const interval = setInterval(() => {
      refreshBlockStatus(false); // Silent background polling - no loading indicator
    }, 30000); // 30s cadence - reduced from 10s to minimize re-renders

    return () => clearInterval(interval);
  }, [refreshBlockStatus]);

  // Mock Fallback for Testing (if API fails or returns incomplete data)
  const mockIncident = {
    id: incidentId || 123,
    status: 'open',
    risk_score: 85,
    src_ip: '45.142.212.61',
    created_at: new Date().toISOString(),
    triage_note: {
      rationale: ['Known Malicious IP', 'Brute Force Pattern', 'High Volume Traffic'],
      summary: "Multiple failed SSH login attempts detected from a known malicious IP address. The traffic pattern matches distinct brute-force signatures.",
      recommendation: "Immediate IP Block and Firewall Rule Update"
    },
    agent_actions: [
      {
        id: "act-001",
        name: "Analyzed IP Reputation",
        description: "IP found in AbuseIPDB with 100% confidence score.",
        status: "completed",
        timestamp: new Date(Date.now() - 1000 * 300).toISOString(),
        agent: "Hunter",
        scope: { type: 'entity', target: '45.142.212.61' },
        riskLevel: 'read-only',
        execution_method: "automated",
        expectedImpact: "None (Read-only)",
        requiresApproval: false,
        estimatedDuration: "2s"
      },
      {
        id: "act-002",
        name: "Correlated with Threat Feeds",
        description: "Matched known C2 server list from AlienVault OTX.",
        status: "completed",
        timestamp: new Date(Date.now() - 1000 * 240).toISOString(),
        agent: "Hunter",
        scope: { type: 'global', target: 'Threat Feeds' },
        riskLevel: 'read-only',
        execution_method: "automated",
        expectedImpact: "None (Read-only)",
        requiresApproval: false,
        estimatedDuration: "5s"
      },
      {
        id: "act-003",
        name: "Checked Active Sessions",
        description: "No active established sessions found for source IP.",
        status: "completed",
        timestamp: new Date(Date.now() - 1000 * 180).toISOString(),
        agent: "Containment",
        scope: { type: 'entity', target: '45.142.212.61' },
        riskLevel: 'read-only',
        execution_method: "automated",
        expectedImpact: "None (Read-only)",
        requiresApproval: false,
        estimatedDuration: "1s"
      },
      {
        id: "act-004",
        name: "IP Blocking",
        description: "Blocked malicious IP address on firewall.",
        status: "completed",
        timestamp: new Date(Date.now() - 1000 * 60).toISOString(),
        agent: "Containment",
        scope: { type: 'entity', target: '45.142.212.61' },
        riskLevel: 'high',
        execution_method: "automated",
        expectedImpact: "Traffic from IP will be dropped",
        requiresApproval: false,
        estimatedDuration: "5s",
        rollback_id: "rb-004"
      },
    ],
    detailed_events: Array.from({ length: 20 }).map((_, i) => ({
      ts: new Date(Date.now() - 1000 * 60 * i).toISOString(),
      eventid: i % 3 === 0 ? 'SSH_FAIL' : 'NET_FLOW',
      message: i % 3 === 0 ? `Failed password for invalid user admin from ${'45.142.212.61'} port ${45000 + i} ssh2` : `Connection attempt from ${'45.142.212.61'}`
    }))
  };

  const activeIncident = (!loading && incident) ? incident : mockIncident;

  // Auto-refresh AI analysis when event count changes
  React.useEffect(() => {
    const currentEventCount = activeIncident.events_analyzed_count || activeIncident.triage_note?.event_count || 0;
    if (currentEventCount > 0 && currentEventCount !== lastEventCountRef.current) {
      // New events detected, refresh analysis
      fetchAiAnalysis();
    }
  }, [activeIncident.events_analyzed_count, activeIncident.triage_note?.event_count, fetchAiAnalysis]);

  // Correctly format risk score (0-1 to 0-100 if needed)
  // Use the HIGHER of risk_score or ml_confidence - threat score should reflect actual threat level
  const riskScore = Math.max(
    activeIncident.risk_score || 0,
    activeIncident.ml_confidence || 0,
    activeIncident.containment_confidence || 0
  );
  const displayScore = riskScore <= 1 ? Math.round(riskScore * 100) : Math.round(riskScore);

  // Determine if AI or prior actions blocked the source IP
  // Check the MOST RECENT block/unblock action to determine actual state
  const { agentBlockedByHistory, manualBlockedByHistory } = React.useMemo(() => {
    // Sort all actions by timestamp to find the most recent block/unblock
    const allActions = [
      ...(activeIncident.agent_actions || []).map((a: any) => ({ ...a, source: 'agent' })),
      ...(activeIncident.actions || []).map((a: any) => ({ ...a, source: 'manual' }))
    ].filter((action: any) => {
      const name = `${action.action || action.action_name || ""}`.toLowerCase();
      const status = `${action.status || action.result || ""}`.toLowerCase();
      return name.includes("block") && name.includes("ip") && (status.includes("success") || status.includes("complete"));
    }).sort((a: any, b: any) => {
      const timeA = new Date(a.executed_at || a.timestamp || a.created_at || 0).getTime();
      const timeB = new Date(b.executed_at || b.timestamp || b.created_at || 0).getTime();
      return timeB - timeA; // Most recent first
    });

    if (allActions.length === 0) {
      return { agentBlockedByHistory: false, manualBlockedByHistory: false };
    }

    // The most recent action determines the state
    const mostRecent = allActions[0];
    const name = `${mostRecent.action || mostRecent.action_name || ""}`.toLowerCase();
    const isUnblock = name.includes("unblock");

    // If most recent action is unblock, not blocked; if block, blocked
    if (isUnblock) {
      return { agentBlockedByHistory: false, manualBlockedByHistory: false };
    }

    return {
      agentBlockedByHistory: mostRecent.source === 'agent',
      manualBlockedByHistory: mostRecent.source === 'manual'
    };
  }, [activeIncident.agent_actions, activeIncident.actions]);

  // PRIORITIZE blockStatus from API (real-time check) over action history
  const isSourceBlocked = blockStatus !== null
    ? blockStatus.is_blocked
    : (agentBlockedByHistory || manualBlockedByHistory);

  const blockBadgeText = isCheckingBlock
    ? "Checking status..."
    : isSourceBlocked
      ? agentBlockedByHistory
        ? "Blocked by AI agent"
        : "Blocked"
      : undefined;

  const handleBlockIp = async () => {
    if (!incidentId) return;
    setBlockActionLoading("block");
    try {
      const result = await socBlockIP(incidentId);
      alert(result?.message || "Action executed: Block IP");
      await refreshIncident();
      await refreshBlockStatus();
    } catch (error) {
      console.error("Failed to block IP:", error);
      alert("Failed to block IP");
    } finally {
      setBlockActionLoading(null);
    }
  };

  const handleUnblockIp = async () => {
    if (!incidentId) return;
    setBlockActionLoading("unblock");
    try {
      const result = await socUnblockIP(incidentId);
      alert(result?.message || "Action executed: Unblock IP");
      await refreshIncident();
      await refreshBlockStatus();
    } catch (error) {
      console.error("Failed to unblock IP:", error);
      alert("Failed to unblock IP");
    } finally {
      setBlockActionLoading(null);
    }
  };

  // Consolidate Action History Data
  // Merge real manual actions, advanced workflow actions, and agent actions
  const allActions = [
    ...(activeIncident.agent_actions || []).map((a: any) => ({
      ...a,
      agent: a.agent || a.agent_type || 'AI Agent',
      // Ensure compatibility with new Action interface
      scope: a.scope || { type: 'entity', target: activeIncident.src_ip },
      riskLevel: a.riskLevel || 'low',
      expectedImpact: a.expectedImpact || 'Standard execution',
      requiresApproval: a.requiresApproval || false,
      estimatedDuration: a.estimatedDuration || 'N/A'
    })),
    ...(activeIncident.actions || []).map((a: any) => ({
      ...a,
      agent_type: null
    })),
    ...(activeIncident.advanced_actions || []).map((a: any) => ({
      ...a,
      agent_type: 'playbook',
      // Ensure workflow actions map fields correctly for the sheet
      action: a.action_name || a.action_type,
      detail: a.result_data ? JSON.stringify(a.result_data) : (a.detail || "Workflow execution")
    }))
  ].sort((a, b) => {
    const dateA = new Date(a.timestamp || a.created_at || Date.now()).getTime();
    const dateB = new Date(b.timestamp || b.created_at || Date.now()).getTime();
    return dateB - dateA;
  });

  if (loading && !incident) {
    // Fallback for initial load only
    return (
      <div className="h-screen w-full flex items-center justify-center bg-background text-foreground">
        <div className="flex flex-col items-center gap-4">
          <Activity className="w-12 h-12 animate-pulse text-primary" />
          <p className="text-muted-foreground">Loading Incident Intelligence...</p>
        </div>
      </div>
    );
  }

  const timeActive = "2h 15m";

  // Extract risk factors from triage note and indicators
  const riskFactors = [];

  // Add recommendation as a risk factor
  if (activeIncident.triage_note?.recommendation) {
    riskFactors.push({
      label: activeIncident.triage_note.recommendation,
      score: Math.round((activeIncident.triage_note.confidence || 0) * 100),
      type: 'negative' as const
    });
  }

  // Add summary as context
  if (activeIncident.triage_note?.summary) {
    riskFactors.push({
      label: activeIncident.triage_note.summary,
      score: Math.round((activeIncident.triage_note.anomaly_score || 0) * 100),
      type: 'negative' as const
    });
  }

  // Add council reasoning if available
  if (activeIncident.council_reasoning) {
    riskFactors.push({
      label: activeIncident.council_reasoning,
      score: 90,
      type: 'negative' as const
    });
  }

  return (
    <div className="h-screen w-full bg-background text-foreground flex flex-col overflow-hidden relative">

      {/* Action History Sheet */}
      <ActionHistorySheet
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        actions={allActions}
        onRollback={(id) => alert(`Rollback requested for action ${id}`)}
      />

      {/* Entity Action History Sheet */}
      <EntityActionHistory
        isOpen={isEntityActionsOpen}
        onClose={() => setIsEntityActionsOpen(false)}
        entityId={activeIncident.src_ip}
        entityType="External IP"
        actions={allActions}
        onRollback={async (rollbackId) => {
          // Find the action to rollback
          const actionToRollback = allActions.find(a => a.rollback_id === rollbackId);

          if (!actionToRollback) {
            alert("Unable to find action to rollback");
            return;
          }

          // Determine action type and execute appropriate rollback
          const actionName = (actionToRollback.name || actionToRollback.action || "").toLowerCase();

          if (actionName.includes('block') && actionName.includes('ip')) {
            // IP Blocking rollback = Unblock IP
            try {
              await handleUnblockIp();
              alert(`Successfully unblocked IP: ${activeIncident.src_ip}`);
            } catch (error) {
              console.error("Rollback failed:", error);
              alert("Failed to unblock IP. Check console for details.");
            }
          } else {
            // Generic rollback - placeholder for other action types
            alert(`Rollback requested for: ${actionToRollback.name || actionToRollback.action}\nRollback ID: ${rollbackId}\n\n(Backend integration pending for this action type)`);
          }
        }}
      />

      {/* Agent Capabilities Sheet */}
      <AgentCapabilitiesSheet
        isOpen={isCapabilitiesOpen}
        onClose={() => setIsCapabilitiesOpen(false)}
        incidentId={incidentId}
        sourceIp={activeIncident.src_ip}
        targetHost={activeIncident.dst_host || activeIncident.target_host}
        onActionComplete={async (result) => {
          console.log('Action completed:', result);
          // Refresh incident data to reflect the action
          await refreshIncident();
          await refreshBlockStatus();
        }}
        onActionError={(result) => {
          console.error('Action failed:', result);
        }}
      />

      {/* Selected Agent Action Detail (if clicked from list) */}
      {selectedAgentAction && (
        <ActionHistorySheet
          isOpen={!!selectedAgentAction}
          onClose={() => setSelectedAgentAction(null)}
          actions={[selectedAgentAction]}
        />
      )}

      {/* Deep Analysis Sheet */}
      <DeepAnalysisSheet
        isOpen={isDeepAnalysisOpen}
        onClose={() => setIsDeepAnalysisOpen(false)}
        incident={activeIncident}
        coordination={coordination}
        coordinationLoading={coordinationLoading}
      />

      {/* Events History Sheet */}
      <EventsHistorySheet
        isOpen={isEventsOpen}
        onClose={() => setIsEventsOpen(false)}
        events={activeIncident.detailed_events || []}
        incidentId={activeIncident.id}
      />

      {/* A. Global Command Header */}
      <header className="border-b bg-card px-6 py-3 flex items-center justify-between shrink-0 z-20 shadow-sm">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => router.push('/incidents')} className="mr-2">
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-xl font-bold flex items-center gap-3">
                Incident #{activeIncident.id}
                <Badge variant={activeIncident.status === 'open' ? 'destructive' : activeIncident.status === 'investigating' ? 'secondary' : 'outline'} className="uppercase">
                  {activeIncident.status}
                </Badge>
                {activeIncident.auto_contained && (
                  <Badge variant="default" className="bg-green-600 text-white gap-1">
                    <Shield className="w-3 h-3" />
                    Auto-Contained
                  </Badge>
                )}
              </h1>
              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                <Badge variant="outline" className="text-[10px] h-5 px-1.5 font-mono">{activeIncident.src_ip}</Badge>
                <span>•</span>
                <span suppressHydrationWarning>Created {new Date(activeIncident.created_at).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-muted/30 px-4 py-1.5 rounded-full border border-border/50">
          <Clock className="w-4 h-4 text-orange-500" />
          <span className="text-sm font-medium font-mono">{timeActive}</span>
          <span className="text-xs text-muted-foreground">Time Active</span>
        </div>

        <div className="flex items-center gap-3">
          {/* Global Copilot Trigger */}
          <Button
            variant={isCopilotOpen ? "default" : "outline"}
            size="sm"
            className="gap-2 mr-2 hidden md:flex"
            onClick={() => setIsCopilotOpen(!isCopilotOpen)}
          >
            <Bot className="w-4 h-4" />
            Copilot
          </Button>

          <Separator orientation="vertical" className="h-8" />

          <div className="flex items-center gap-2 mr-2">
            <Avatar className="h-8 w-8 border">
              <AvatarFallback>AI</AvatarFallback>
            </Avatar>
            <div className="flex flex-col text-xs hidden lg:flex">
              <span className="font-medium">System Assignee</span>
              <span className="text-muted-foreground">Auto-Pilot</span>
            </div>
          </div>

          <Button variant="destructive" size="sm" className="gap-2">
            <AlertTriangle className="w-4 h-4" />
            Escalate
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <CheckCircle className="w-4 h-4" />
            Close
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreVertical className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setIsHistoryOpen(true)}>
                <History className="w-4 h-4 mr-2" />
                View Action History
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Share2 className="w-4 h-4 mr-2" />
                Share Incident
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="flex-1 grid grid-cols-[320px_minmax(0,1fr)_350px] overflow-hidden divide-x divide-border">

        {/* B. Column 1: Context & Intelligence */}
        <aside className="h-full overflow-y-auto bg-card/50 p-4 flex flex-col gap-4">
          {/* AI Threat Score with Primary Entity */}
          <div className="space-y-2">
            <ThreatScoreCard
              score={displayScore}
              factors={riskFactors}
              entityIp={activeIncident.src_ip}
              entityType="External IP"
              onOpenEntityActions={() => setIsEntityActionsOpen(true)}
            />
          </div>

          {/* Enhanced AI Analysis Section */}
          <div className="flex-1 min-h-0 flex flex-col">
            <AIAnalysisCard
              triageNote={aiAnalysis?.triage_note || activeIncident.triage_note}
              analysis={aiAnalysis?.analysis}
              isLoading={aiAnalysisLoading}
              onRefresh={() => fetchAiAnalysis(true)}
              onShowDeepAnalysis={() => setIsDeepAnalysisOpen(true)}
              className="flex-1"
            />
          </div>
        </aside>

        {/* C. Column 2: Investigation Workspace */}
        <InvestigationWorkspace
          activeIncident={activeIncident}
          allActions={allActions}
          onActionClick={setSelectedAgentAction}
          onHistoryOpen={() => setIsHistoryOpen(true)}
          onEventsOpen={() => setIsEventsOpen(true)}
        />

        {/* D. Column 3: Response & Automation */}
        <ResponseSection
          activeIncident={activeIncident}
          isSourceBlocked={isSourceBlocked}
          blockBadgeText={blockBadgeText}
          isCheckingBlock={isCheckingBlock}
          blockActionLoading={blockActionLoading}
          onBlockIp={handleBlockIp}
          onUnblockIp={handleUnblockIp}
          onOpenMoreActions={() => setIsCapabilitiesOpen(true)}
          onOpenEntityActions={() => setIsEntityActionsOpen(true)}
          isCopilotOpen={isCopilotOpen}
          setIsCopilotOpen={setIsCopilotOpen}
        />
      </main>
    </div>
  );
}
