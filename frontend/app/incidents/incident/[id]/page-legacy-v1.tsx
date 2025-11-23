"use client";

import React, { useState, useEffect, use, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2, Shield, Search, AlertOctagon, FileText, MessageSquare, Workflow, TrendingUp, Activity, Clock, CheckCircle, AlertTriangle } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useIncidentRealtime } from "@/app/hooks/useIncidentRealtime";
import { apiUrl } from "@/app/utils/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import EnhancedAIAnalysis from "@/components/EnhancedAIAnalysis";
import UnifiedResponseTimeline from "@/components/UnifiedResponseTimeline";
import ThreatStatusBar from "@/components/ThreatStatusBar";

interface IncidentDetail {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  escalation_level?: string;
  risk_score?: number;
  threat_category?: string;
  containment_confidence?: number;
  containment_method?: string;
  agent_id?: string;
  agent_actions?: Array<{action: string; status: string}>;
  agent_confidence?: number;
  ml_features?: Record<string, unknown>;
  ensemble_scores?: Record<string, number>;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
  actions: Array<any>;
  advanced_actions?: Array<any>;
  detailed_events: Array<any>;
  iocs: {
    ip_addresses: string[];
    domains: string[];
    hashes: string[];
  };
}

export default function EnterpriseIncidentPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const { id } = use(params);
  const incidentId = parseInt(id);

  const {
    incident,
    loading: realtimeLoading,
    refreshIncident,
    connectionStatus,
    lastUpdate
  } = useIncidentRealtime({
    incidentId,
    autoRefresh: true,
    refreshInterval: 5000
  });

  const [executing, setExecuting] = useState(false);

  const fetchIncident = useCallback(async () => {
    return await refreshIncident();
  }, [refreshIncident]);

  useEffect(() => {
    fetchIncident();
  }, [incidentId]);

  const handleExecuteRecommendation = async (action: string, params?: Record<string, any>) => {
    try {
      setExecuting(true);
      const response = await fetch(
        apiUrl(`/api/incidents/${incidentId}/execute-ai-recommendation`),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
          },
          body: JSON.stringify({ action_type: action, parameters: params || {} })
        }
      );

      const data = await response.json();

      if (response.ok && data.success) {
        await fetchIncident();
        alert(`✅ Action executed successfully!\n\n${data.action_name}\n\nDetails: ${data.result?.detail || 'Completed'}`);
      } else {
        // Extract more detailed error information
        let errorMessage = 'Execution failed';

        if (data.error) {
          errorMessage = data.error;
        } else if (data.detail) {
          errorMessage = data.detail;
        } else if (data.result?.detail) {
          errorMessage = data.result.detail;
        }

        throw new Error(errorMessage);
      }
    } catch (err) {
      console.error('Failed to execute recommendation:', err);
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      alert(`❌ Failed to execute action\n\nError: ${errorMsg}\n\nPlease check:\n- T-Pot connection status\n- SSH credentials are configured\n- Firewall access from your IP\n- Backend logs for details`);
    } finally {
      setExecuting(false);
    }
  };

  const handleExecuteAllRecommendations = async () => {
    if (!confirm('Execute all AI-recommended priority actions?\n\nThis will create a workflow with multiple automated responses.')) {
      return;
    }

    try {
      setExecuting(true);
      const response = await fetch(
        apiUrl(`/api/incidents/${incidentId}/execute-ai-plan`),
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key'
          }
        }
      );

      const data = await response.json();
      if (data.success) {
        await fetchIncident();
        alert(`AI Plan Executed!\n\n${data.successful_actions} actions succeeded\n${data.failed_actions} actions failed`);
      } else {
        throw new Error(data.error || 'Execution failed');
      }
    } catch (err) {
      console.error('Failed to execute AI plan:', err);
      alert(`Failed to execute AI plan: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setExecuting(false);
    }
  };

  const handleContainNow = async () => {
    if (!incident) return;
    await handleExecuteRecommendation('block_ip', { ip: incident.src_ip, duration: 30 });
  };

  const handleHuntThreats = async () => {
    if (!incident) return;
    await handleExecuteRecommendation('hunt_similar_attacks', {});
  };

  const handleRollback = async (rollbackId: string) => {
    try {
      const response = await fetch(
        apiUrl(`/api/agents/actions/rollback/${rollbackId}`),
        {
          method: 'POST',
          headers: { 'x-api-key': process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key' }
        }
      );
      if (response.ok) await fetchIncident();
      else throw new Error('Rollback failed');
    } catch (err) {
      console.error('Rollback failed:', err);
      alert('Failed to rollback action');
    }
  };

  if (realtimeLoading && !incident) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto" />
          <p className="text-muted-foreground">Loading incident details...</p>
        </div>
      </div>
    );
  }

  if (!incident) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <p className="text-destructive text-lg font-semibold">Incident not found</p>
          <Button onClick={() => router.push('/incidents')}>Back to Incidents</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="border-b bg-card sticky top-0 z-10 shadow-sm">
        <div className="max-w-[1600px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="icon" onClick={() => router.push('/incidents')}>
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <div>
                <h1 className="text-2xl font-bold flex items-center gap-3">
                  Incident #{incident.id}
                  <Badge variant={
                    incident.status === 'open' ? 'destructive' :
                    incident.status === 'investigating' ? 'secondary' :
                    incident.status === 'contained' ? 'default' : 'outline'
                  } className="uppercase">
                    {incident.status}
                  </Badge>
                </h1>
                <div className="flex items-center gap-3 text-sm text-muted-foreground mt-1">
                  <span className="font-mono bg-muted px-1.5 rounded text-foreground">{incident.src_ip}</span>
                  <span>•</span>
                  <span>{new Date(incident.created_at).toLocaleString()}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' :
                  connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-muted-foreground">
                  {connectionStatus === 'connected' ? 'Live' : connectionStatus === 'connecting' ? 'Connecting...' : 'Offline'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[1600px] mx-auto px-6 py-6 space-y-6">
        {/* Threat Status Bar */}
        <Card className="border-none bg-transparent shadow-none p-0">
            <ThreatStatusBar incident={incident} />
        </Card>

        {/* Tactical Decision Center */}
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    Tactical Decision Center
                </CardTitle>
                <CardDescription>Immediate response actions and playbook execution</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <Button className="h-auto py-4 flex flex-col gap-2" variant="destructive" onClick={handleContainNow}>
                    <Shield className="w-5 h-5" />
                    <span className="text-xs font-semibold">Contain Now</span>
                </Button>
                <Button className="h-auto py-4 flex flex-col gap-2" variant="secondary" onClick={handleHuntThreats}>
                    <Search className="w-5 h-5" />
                    <span className="text-xs font-semibold">Hunt Threats</span>
                </Button>
                <Button className="h-auto py-4 flex flex-col gap-2" variant="outline" onClick={() => alert('Escalating...')}>
                    <AlertOctagon className="w-5 h-5" />
                    <span className="text-xs font-semibold">Escalate</span>
                </Button>
                <Button className="h-auto py-4 flex flex-col gap-2" variant="outline" onClick={() => alert('Creating Playbook...')}>
                    <Workflow className="w-5 h-5" />
                    <span className="text-xs font-semibold">Create Playbook</span>
                </Button>
                <Button className="h-auto py-4 flex flex-col gap-2" variant="outline" onClick={() => alert('Generating Report...')}>
                    <FileText className="w-5 h-5" />
                    <span className="text-xs font-semibold">Generate Report</span>
                </Button>
                <Button className="h-auto py-4 flex flex-col gap-2" variant="outline" onClick={() => alert('Opening AI Chat...')}>
                    <MessageSquare className="w-5 h-5" />
                    <span className="text-xs font-semibold">Ask AI</span>
                </Button>
            </CardContent>
        </Card>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: AI Analysis */}
          <Card className="h-full flex flex-col">
            <CardHeader>
                <CardTitle>AI Analysis & Recommendations</CardTitle>
            </CardHeader>
            <CardContent className="flex-1">
                <EnhancedAIAnalysis
                  incident={incident}
                  onExecuteRecommendation={handleExecuteRecommendation}
                  onExecuteAllRecommendations={handleExecuteAllRecommendations}
                />
            </CardContent>
          </Card>

          {/* Right: Response Timeline */}
          <Card className="h-full flex flex-col">
             <CardHeader>
                <CardTitle>Response Timeline</CardTitle>
             </CardHeader>
             <CardContent className="flex-1 p-0">
                <ScrollArea className="h-[600px] px-6 pb-6">
                    <UnifiedResponseTimeline
                      incidentId={incident.id}
                      actions={incident.actions}
                      automatedActions={incident.advanced_actions}
                      onRefresh={fetchIncident}
                      onRollback={handleRollback}
                      incidentEvents={incident.detailed_events}
                    />
                </ScrollArea>
             </CardContent>
          </Card>
        </div>

        {/* Bottom: Detailed Tabs */}
        <Card>
            <CardContent className="p-6">
                <Tabs defaultValue="timeline" className="w-full">
                    <TabsList className="w-full justify-start border-b rounded-none h-auto p-0 bg-transparent space-x-6">
                        <TabsTrigger value="timeline" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent pb-3">Attack Timeline</TabsTrigger>
                        <TabsTrigger value="iocs" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent pb-3">IOCs & Evidence</TabsTrigger>
                        <TabsTrigger value="ml" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent pb-3">ML Analysis</TabsTrigger>
                    </TabsList>

                    <TabsContent value="timeline" className="mt-6">
                        {incident.detailed_events?.length > 0 ? (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead className="w-[180px]">Timestamp</TableHead>
                                        <TableHead>Event ID</TableHead>
                                        <TableHead>Message</TableHead>
                                        <TableHead>Source</TableHead>
                                        <TableHead>Destination</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {incident.detailed_events.slice(0, 20).map((event: any, idx: number) => (
                                        <TableRow key={idx}>
                                            <TableCell className="text-muted-foreground text-xs">{new Date(event.ts).toLocaleString()}</TableCell>
                                            <TableCell className="font-mono text-xs">{event.eventid}</TableCell>
                                            <TableCell>{event.message}</TableCell>
                                            <TableCell className="font-mono text-xs">{event.src_ip}</TableCell>
                                            <TableCell className="font-mono text-xs">{event.dst_ip}:{event.dst_port}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        ) : (
                            <div className="text-center py-12 text-muted-foreground">No detailed events available</div>
                        )}
                    </TabsContent>

                    <TabsContent value="iocs" className="mt-6 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">IP Addresses</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {incident.iocs?.ip_addresses?.length > 0 ? (
                                        <div className="space-y-1">
                                            {incident.iocs.ip_addresses.map((ip: string, idx: number) => (
                                                <div key={idx} className="text-sm font-mono bg-muted/50 p-1 px-2 rounded">{ip}</div>
                                            ))}
                                        </div>
                                    ) : <p className="text-sm text-muted-foreground">None detected</p>}
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">Domains</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {incident.iocs?.domains?.length > 0 ? (
                                        <div className="space-y-1">
                                            {incident.iocs.domains.map((domain: string, idx: number) => (
                                                <div key={idx} className="text-sm font-mono bg-muted/50 p-1 px-2 rounded">{domain}</div>
                                            ))}
                                        </div>
                                    ) : <p className="text-sm text-muted-foreground">None detected</p>}
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">File Hashes</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {incident.iocs?.hashes?.length > 0 ? (
                                        <div className="space-y-1">
                                            {incident.iocs.hashes.map((hash: string, idx: number) => (
                                                <div key={idx} className="text-xs font-mono bg-muted/50 p-1 px-2 rounded truncate" title={hash}>{hash}</div>
                                            ))}
                                        </div>
                                    ) : <p className="text-sm text-muted-foreground">None detected</p>}
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>

                    <TabsContent value="ml" className="mt-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                             {incident.ensemble_scores && (
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-sm">Ensemble Model Scores</CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-3">
                                        {Object.entries(incident.ensemble_scores).map(([model, score]: [string, any]) => (
                                            <div key={model} className="flex items-center justify-between">
                                                <span className="text-sm font-medium">{model}</span>
                                                <Badge variant="secondary">
                                                    {(typeof score === 'number' ? score * 100 : 0).toFixed(1)}%
                                                </Badge>
                                            </div>
                                        ))}
                                    </CardContent>
                                </Card>
                             )}
                             {incident.ml_features && (
                                 <Card>
                                     <CardHeader>
                                         <CardTitle className="text-sm">ML Features</CardTitle>
                                     </CardHeader>
                                     <CardContent>
                                         <ScrollArea className="h-[200px]">
                                            <div className="space-y-2">
                                                {Object.entries(incident.ml_features).map(([key, value]: [string, any]) => (
                                                    <div key={key} className="flex justify-between text-sm border-b border-border/50 pb-1 last:border-0">
                                                        <span className="text-muted-foreground">{key}</span>
                                                        <span className="font-mono">{String(value)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                         </ScrollArea>
                                     </CardContent>
                                 </Card>
                             )}
                        </div>
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Card>
      </div>

      {/* Executing Overlay */}
      {executing && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-card border shadow-lg rounded-xl p-8 text-center max-w-sm">
            <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Executing Action</h3>
            <p className="text-muted-foreground text-sm">Please wait while the system processes your request...</p>
          </div>
        </div>
      )}
    </div>
  );
}
