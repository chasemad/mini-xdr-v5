"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Loader2, Target, Box, Search, Eye, Clock, CheckCircle2 } from "lucide-react";
import type { IncidentCoordination } from "@/lib/agent-api";

interface AIAgentsTabProps {
  coordination: IncidentCoordination | null;
  loading: boolean;
}

export function AIAgentsTab({ coordination, loading }: AIAgentsTabProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-muted-foreground">Syncing with Agent Coordination Hub...</span>
      </div>
    );
  }

  if (!coordination) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No coordination data available.
      </div>
    );
  }

  const decisions = coordination.agent_decisions || {};
  const attribution = decisions.attribution;
  const containment = decisions.containment;
  const forensics = decisions.forensics;
  const deception = decisions.deception;

  return (
    <div className="space-y-6">
        {/* Coordination Status */}
        <Card>
            <CardHeader>
                <div className="flex justify-between items-center">
                    <CardTitle className="flex items-center gap-2">
                        Agent Coordination Hub
                        <Badge variant="outline">{coordination.participating_agents?.length || 0} Agents Active</Badge>
                    </CardTitle>
                    <Badge className="bg-green-500">
                        {coordination.coordination_status || "Active"}
                    </Badge>
                </div>
            </CardHeader>
            <CardContent>
                {coordination.recommendations?.length > 0 && (
                    <Alert className="bg-primary/5 border-primary/10 mb-4">
                        <AlertTitle className="mb-2 font-semibold text-primary">Strategic Recommendations</AlertTitle>
                        <ul className="list-disc pl-4 space-y-1 text-sm">
                            {coordination.recommendations.map((rec, i) => (
                                <li key={i}>{rec}</li>
                            ))}
                        </ul>
                    </Alert>
                )}
            </CardContent>
        </Card>

        {/* Agent Cards Grid */}
        <div className="grid md:grid-cols-2 gap-4">
            {/* Attribution Agent */}
            <Card className={attribution ? "" : "opacity-50"}>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Target className="h-4 w-4 text-red-500" />
                        Attribution Agent
                    </CardTitle>
                </CardHeader>
                <CardContent className="text-sm">
                    {attribution ? (
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Threat Actor</span>
                                <span className="font-mono font-bold">{attribution.threat_actor || "Unknown"}</span>
                            </div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Confidence</span>
                                    <span>{Math.round(attribution.confidence * 100)}%</span>
                                </div>
                                <Progress value={attribution.confidence * 100} className="h-1" />
                            </div>
                            <div>
                                <span className="text-muted-foreground text-xs">TTPs Identified</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                    {attribution.tactics?.map((t: string) => (
                                        <Badge key={t} variant="secondary" className="text-[10px]">{t}</Badge>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : <p className="text-muted-foreground text-xs">Waiting for assignment...</p>}
                </CardContent>
            </Card>

            {/* Containment Agent */}
            <Card className={containment ? "" : "opacity-50"}>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Box className="h-4 w-4 text-blue-500" />
                        Containment Agent
                    </CardTitle>
                </CardHeader>
                <CardContent className="text-sm">
                    {containment ? (
                        <div className="space-y-3">
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Status</span>
                                <Badge variant={containment.status === "active" ? "destructive" : "outline"}>
                                    {containment.status}
                                </Badge>
                            </div>
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs">
                                    <span>Effectiveness</span>
                                    <span>{Math.round(containment.effectiveness * 100)}%</span>
                                </div>
                                <Progress value={containment.effectiveness * 100} className="h-1" />
                            </div>
                            <div>
                                <span className="text-muted-foreground text-xs">Actions Taken</span>
                                <ul className="list-disc pl-4 mt-1 text-xs">
                                    {containment.actions_taken?.map((a: string) => (
                                        <li key={a}>{a}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ) : <p className="text-muted-foreground text-xs">Waiting for assignment...</p>}
                </CardContent>
            </Card>

             {/* Forensics Agent */}
            <Card className={forensics ? "" : "opacity-50"}>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Search className="h-4 w-4 text-amber-500" />
                        Forensics Agent
                    </CardTitle>
                </CardHeader>
                <CardContent className="text-sm">
                    {forensics ? (
                        <div className="space-y-3">
                            <div className="grid grid-cols-2 gap-2 text-xs text-center">
                                <div className="bg-muted p-2 rounded">
                                    <div className="font-bold">{forensics.evidence_collected?.length || 0}</div>
                                    <div className="text-muted-foreground">Artifacts</div>
                                </div>
                                <div className="bg-muted p-2 rounded">
                                    <div className="font-bold">{forensics.timeline_events || 0}</div>
                                    <div className="text-muted-foreground">Events</div>
                                </div>
                            </div>
                            <div className="text-xs">
                                <span className="text-muted-foreground">Suspicious Processes:</span> {forensics.suspicious_processes || 0}
                            </div>
                        </div>
                    ) : <p className="text-muted-foreground text-xs">Waiting for assignment...</p>}
                </CardContent>
            </Card>

             {/* Deception Agent */}
            <Card className={deception ? "" : "opacity-50"}>
                <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                        <Eye className="h-4 w-4 text-purple-500" />
                        Deception Agent
                    </CardTitle>
                </CardHeader>
                <CardContent className="text-sm">
                    {deception ? (
                        <div className="space-y-3">
                             <div className="flex justify-between">
                                <span className="text-muted-foreground">Honeytokens</span>
                                <span className="font-mono font-bold">{deception.honeytokens_deployed || 0}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Interactions</span>
                                <span className="font-mono font-bold text-red-500">{deception.attacker_interactions || 0}</span>
                            </div>
                        </div>
                    ) : <p className="text-muted-foreground text-xs">Inactive / Not Deployed</p>}
                </CardContent>
            </Card>
        </div>

        {/* Timeline Preview */}
        <Card>
            <CardHeader className="pb-2">
                 <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Coordination Timeline
                 </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4 relative pl-4 border-l-2 border-muted my-2">
                    {coordination.coordination_timeline?.slice(0, 5).map((event, idx) => (
                        <div key={idx} className="relative">
                            <div className="absolute -left-[21px] top-0 h-3 w-3 rounded-full bg-background border-2 border-primary" />
                            <div className="text-xs font-semibold">{event.event}</div>
                            <div className="text-[10px] text-muted-foreground" suppressHydrationWarning>
                                {new Date(event.timestamp).toLocaleTimeString()} • {event.details}
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    </div>
  );
}
