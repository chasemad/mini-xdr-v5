"use client";

import React, { use, useState } from "react";
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

import { agentApi, type IncidentCoordination } from "@/lib/agent-api";
import DeepAnalysisSheet from "@/components/v2/DeepAnalysisSheet";
import AIAnalysisCard from "@/components/v2/AIAnalysisCard";
import ResponseSection from "@/components/v2/ResponseSection";
import InvestigationWorkspace from "@/components/v2/InvestigationWorkspace";
import { getBlockStatus, socBlockIP, socUnblockIP } from "@/app/lib/api";

export default function IncidentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const incidentId = parseInt(id);
  const router = useRouter();
  const [isAgentActionsOpen, setIsAgentActionsOpen] = useState(true);
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isCapabilitiesOpen, setIsCapabilitiesOpen] = useState(false);
  const [isDeepAnalysisOpen, setIsDeepAnalysisOpen] = useState(false);
  const [selectedAgentAction, setSelectedAgentAction] = useState<any>(null);
  const [blockStatus, setBlockStatus] = useState<{ is_blocked: boolean; ip?: string; last_checked?: string } | null>(null);
  const [isCheckingBlock, setIsCheckingBlock] = useState(false);
  const [blockActionLoading, setBlockActionLoading] = useState<"block" | "unblock" | null>(null);

  const {
    incident,
    loading,
    refreshIncident
  } = useIncidentRealtime({
    incidentId,
    autoRefresh: false,
    refreshInterval: 10000
  });

  const [coordination, setCoordination] = useState<IncidentCoordination | null>(null);
  const [coordinationLoading, setCoordinationLoading] = useState(true);

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
    }
  }, [incidentId]);

  const refreshBlockStatus = React.useCallback(async () => {
    if (!incidentId) return;
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("access_token");
      if (!token) return;
    }
    try {
      setIsCheckingBlock(true);
      const status = await getBlockStatus(incidentId);
      if (status) {
        setBlockStatus(status);
      }
    } catch (error) {
      console.error("Failed to fetch block status:", error);
    } finally {
      setIsCheckingBlock(false);
    }
  }, [incidentId]);

  React.useEffect(() => {
    refreshBlockStatus();
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
      { action: "Analyzed IP Reputation", status: "completed", timestamp: new Date(Date.now() - 1000 * 300).toISOString(), execution_method: "automated", detail: "IP found in AbuseIPDB with 100% confidence score." },
      { action: "Correlated with Threat Feeds", status: "completed", timestamp: new Date(Date.now() - 1000 * 240).toISOString(), execution_method: "automated", detail: "Matched known C2 server list from AlienVault OTX." },
      { action: "Checked Active Sessions", status: "completed", timestamp: new Date(Date.now() - 1000 * 180).toISOString(), execution_method: "automated", detail: "No active established sessions found for source IP." },
    ],
    detailed_events: Array.from({ length: 20 }).map((_, i) => ({
      ts: new Date(Date.now() - 1000 * 60 * i).toISOString(),
      eventid: i % 3 === 0 ? 'SSH_FAIL' : 'NET_FLOW',
      message: i % 3 === 0 ? `Failed password for invalid user admin from ${'45.142.212.61'} port ${45000 + i} ssh2` : `Connection attempt from ${'45.142.212.61'}`
    }))
  };

  const activeIncident = (!loading && incident) ? incident : mockIncident;

  // Correctly format risk score (0-1 to 0-100 if needed)
  const riskScore = activeIncident.risk_score || 0;
  const displayScore = riskScore <= 1 ? Math.round(riskScore * 100) : Math.round(riskScore);

  // Determine if AI or prior actions already blocked the source IP
  const agentBlocked = React.useMemo(() => {
    return (activeIncident.agent_actions || []).some((action: any) => {
      const name = `${action.action || action.action_name || ""}`.toLowerCase();
      const status = `${action.status || action.result || ""}`.toLowerCase();
      return name.includes("block") && name.includes("ip") && (status.includes("success") || status.includes("complete"));
    });
  }, [activeIncident.agent_actions]);

  const manualBlock = React.useMemo(() => {
    return (activeIncident.actions || []).some((action: any) => {
      const name = `${action.action || ""}`.toLowerCase();
      const status = `${action.result || ""}`.toLowerCase();
      return name.includes("block") && name.includes("ip") && (status.includes("success") || status.includes("complete"));
    });
  }, [activeIncident.actions]);

  const isSourceBlocked = blockStatus?.is_blocked || agentBlocked || manualBlock;
  const blockBadgeText = isCheckingBlock
    ? "Checking status..."
    : isSourceBlocked
      ? agentBlocked
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
      agent_type: a.agent_type || 'ai',
      id: a.id || Math.random().toString(36).substr(2, 9)
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

      {/* Agent Capabilities Sheet */}
      <AgentCapabilitiesSheet
        isOpen={isCapabilitiesOpen}
        onClose={() => setIsCapabilitiesOpen(false)}
        onExecute={(actionId) => {
          alert(`Executing: ${actionId}`);
          // In production, this would call the backend API
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
                <span>Created {new Date(activeIncident.created_at).toLocaleString()}</span>
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
      <main className="flex-1 grid grid-cols-[20%_50%_30%] overflow-hidden divide-x divide-border">

        {/* B. Column 1: Context & Intelligence */}
        <aside className="h-full overflow-y-auto bg-card/50 p-4 flex flex-col gap-4">
          {/* AI Threat Score */}
          <div className="space-y-2">
            <ThreatScoreCard
              score={displayScore}
              factors={riskFactors}
            />
          </div>

          {/* Enhanced AI Analysis Section */}
          <div className="flex-1 min-h-0 flex flex-col">
            <AIAnalysisCard
              triageNote={activeIncident.triage_note}
              onShowDeepAnalysis={() => setIsDeepAnalysisOpen(true)}
              className="flex-1"
            />
          </div>

          {/* Entity Card */}
          <Card>
            <CardHeader className="pb-1.5 pt-3 px-4">
              <CardTitle className="text-sm font-medium text-muted-foreground">Primary Entity</CardTitle>
            </CardHeader>
            <CardContent className="pb-3">
              <div className="flex items-center gap-2">
                <div className="p-2 bg-muted rounded-md">
                  <Shield className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <div className="font-mono text-sm font-bold">{activeIncident.src_ip}</div>
                  <div className="text-xs text-muted-foreground">External IP</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </aside>

        {/* C. Column 2: Investigation Workspace */}
        <InvestigationWorkspace
          activeIncident={activeIncident}
          allActions={allActions}
          onActionClick={setSelectedAgentAction}
          onHistoryOpen={() => setIsHistoryOpen(true)}
        />

        {/* D. Column 3: Response & Automation */}
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
          isCopilotOpen={isCopilotOpen}
          setIsCopilotOpen={setIsCopilotOpen}
        />
      </main>
    </div>
  );
}
