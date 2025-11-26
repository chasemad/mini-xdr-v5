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

export default function RedesignedIncidentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const incidentId = parseInt(id);
  const router = useRouter();
  const [isAgentActionsOpen, setIsAgentActionsOpen] = useState(true);
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isCapabilitiesOpen, setIsCapabilitiesOpen] = useState(false);
  const [selectedAgentAction, setSelectedAgentAction] = useState<any>(null);

  const {
    incident,
    loading,
    refreshIncident
  } = useIncidentRealtime({
    incidentId,
    autoRefresh: false,
    refreshInterval: 10000
  });

  // Enhanced mock incident with Phase 2 data
  const mockIncident = {
    id: incidentId,
    status: 'open',
    risk_score: 85,
    src_ip: '45.142.212.61',
    created_at: new Date().toISOString(),
    threat_category: 'malware_botnet',
    escalation_level: 'high',
    ml_confidence: 0.87,
    council_verdict: 'THREAT',
    council_confidence: 0.92,
    routing_path: ['ml_predict', 'council_verify', 'gemini_judge'],
    processing_time_ms: 1250,
    triage_note: {
      rationale: ['Known Malicious IP', 'Brute Force Pattern', 'High Volume Traffic'],
      summary: "Multiple failed SSH login attempts detected from a known malicious IP address. The traffic pattern matches distinct brute-force signatures.",
      recommendation: "Immediate IP Block and Firewall Rule Update",
      confidence: 0.92,
      anomaly_score: 0.85,
      threat_class: 5,
      event_count: 45,
      indicators: {
        enhanced_model_prediction: {
          class_probabilities: [0.01, 0.03, 0.02, 0.04, 0.05, 0.85, 0.00],
          uncertainty_score: 0.15,
          explanation: "High C2 traffic detected",
          feature_importance: {}
        },
        phase2_advanced_features: {
          feature_count: 100,
          features_extracted: true,
          feature_dimensions: "100D (79 base + 21 advanced)"
        }
      },
      council_verified: true
    },
    agent_actions: [
      { action: "Analyzed IP Reputation", status: "completed", timestamp: new Date(Date.now() - 1000 * 300).toISOString(), execution_method: "automated", detail: "IP found in AbuseIPDB with 100% confidence score.", agent_type: 'edr' },
      { action: "Correlated with Threat Feeds", status: "completed", timestamp: new Date(Date.now() - 1000 * 240).toISOString(), execution_method: "automated", detail: "Matched known C2 server list from AlienVault OTX.", agent_type: 'edr' },
      { action: "Checked Active Sessions", status: "completed", timestamp: new Date(Date.now() - 1000 * 180).toISOString(), execution_method: "automated", detail: "No active established sessions found for source IP.", agent_type: 'edr' },
    ],
    detailed_events: Array.from({ length: 20 }).map((_, i) => ({
      ts: new Date(Date.now() - 1000 * 60 * i).toISOString(),
      eventid: i % 3 === 0 ? 'SSH_FAIL' : 'NET_FLOW',
      message: i % 3 === 0 ? `Failed password for invalid user admin from ${'45.142.212.61'} port ${45000+i} ssh2` : `Connection attempt from ${'45.142.212.61'}`,
      src_ip: '45.142.212.61',
      dst_port: i % 3 === 0 ? 45000+i : undefined,
      eventid_num: i % 3 === 0 ? 4001 : 1001,
      raw: {}
    })),
    iocs: {
      ip_addresses: ['45.142.212.61', '192.168.1.100'],
      domains: ['malicious.example.com'],
      hashes: ['a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3']
    }
  };

  const activeIncident = (!loading && incident) ? incident : mockIncident;

  // Mock coordination data
  const mockCoordination = {
    incident_id: incidentId,
    coordination_status: 'council_verified',
    participating_agents: ['attribution', 'containment', 'forensics'],
    agent_decisions: {
      attribution: {
        threat_actor: 'APT29',
        confidence: 0.78,
        tactics: ['initial_access', 'lateral_movement'],
        iocs_identified: 12
      },
      containment: {
        actions_taken: ['isolate_host', 'block_c2'],
        effectiveness: 0.92,
        status: 'active'
      },
      forensics: {
        evidence_collected: ['memory_dump', 'disk_image'],
        timeline_events: 45,
        suspicious_processes: 3
      }
    },
    coordination_timeline: [
      {
        timestamp: new Date(Date.now() - 1000 * 300).toISOString(),
        event: 'council_verification',
        details: 'High confidence malware detection',
        verdict: 'THREAT'
      },
      {
        timestamp: new Date(Date.now() - 1000 * 250).toISOString(),
        event: 'agent_coordination_initiated',
        details: 'Attribution, Containment, and Forensics agents activated',
        agents: ['attribution', 'containment', 'forensics']
      }
    ],
    recommendations: [
      'Immediate containment and forensic analysis required',
      'Block C2 domains: evil.com, malicious.net',
      'Isolate affected systems: host-01, host-02'
    ]
  };

  const activeCoordination = (!coordLoading && coordination) ? coordination : mockCoordination;

  if (loading && !incident) {
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

  return (
    <div className="h-screen w-full bg-background text-foreground flex flex-col overflow-hidden relative">
      {/* Global Command Header */}
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
                {activeIncident.council_verdict && (
                  <Badge variant={activeIncident.council_verdict === 'THREAT' ? 'destructive' : 'outline'} className="uppercase">
                    {activeIncident.council_verdict}
                  </Badge>
                )}
              </h1>
              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                <Badge variant="outline" className="text-[10px] h-5 px-1.5 font-mono">{activeIncident.src_ip}</Badge>
                <span>•</span>
                <span suppressHydrationWarning>Created {new Date(activeIncident.created_at).toLocaleString()}</span>
                {activeIncident.processing_time_ms && (
                  <>
                    <span>•</span>
                    <span>Analysis: {(activeIncident.processing_time_ms / 1000).toFixed(1)}s</span>
                  </>
                )}
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
               <DropdownMenuItem>
                  <Share2 className="w-4 h-4 mr-2" />
                  Share Incident
               </DropdownMenuItem>
               <DropdownMenuItem>
                  <Eye className="w-4 h-4 mr-2" />
                  Full Details
               </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content with Tabs */}
      <main className="flex-1 overflow-hidden">
        <Tabs defaultValue="overview" className="h-full flex flex-col">
          <div className="border-b bg-muted/20 px-6 py-2">
            <TabsList className="grid w-full grid-cols-6">
              <TabsTrigger value="overview" className="text-xs">Overview</TabsTrigger>
              <TabsTrigger value="council" className="text-xs">Council Analysis</TabsTrigger>
              <TabsTrigger value="agents" className="text-xs">AI Agents</TabsTrigger>
              <TabsTrigger value="timeline" className="text-xs">Timeline</TabsTrigger>
              <TabsTrigger value="evidence" className="text-xs">Evidence</TabsTrigger>
              <TabsTrigger value="response" className="text-xs">Response Actions</TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-hidden">
            <TabsContent value="overview" className="h-full m-0">
              <OverviewTab incident={activeIncident} />
            </TabsContent>

            <TabsContent value="council" className="h-full m-0">
              <CouncilAnalysisTab incident={activeIncident} />
            </TabsContent>

            <TabsContent value="agents" className="h-full m-0">
              <AIAgentsTab incident={activeIncident} coordination={activeCoordination} />
            </TabsContent>

            <TabsContent value="timeline" className="h-full m-0">
              <TimelineTab incident={activeIncident} coordination={activeCoordination} />
            </TabsContent>

            <TabsContent value="evidence" className="h-full m-0">
              <EvidenceTab incident={activeIncident} />
            </TabsContent>

            <TabsContent value="response" className="h-full m-0">
              <ResponseActionsTab incident={activeIncident} coordination={activeCoordination} />
            </TabsContent>
          </div>
        </Tabs>
      </main>
    </div>
  );
}
