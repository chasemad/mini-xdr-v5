"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Activity, Shield, Zap, Search } from "lucide-react";
import type { IncidentCoordination } from "@/lib/agent-api";

interface TimelineTabProps {
  incident: any;
  coordination: IncidentCoordination | null;
}

export function TimelineTab({ incident, coordination }: TimelineTabProps) {
  // Merge events from incident details and coordination timeline
  const incidentEvents = incident.detailed_events?.map((e: any) => ({
    timestamp: e.ts,
    type: "event",
    source: e.src_ip,
    message: e.message,
    id: e.eventid
  })) || [];

  const coordinationEvents = coordination?.coordination_timeline?.map((e: any) => ({
    timestamp: e.timestamp,
    type: "coordination",
    source: "Agent Hub",
    message: `${e.event}: ${e.details}`,
    id: "COORD"
  })) || [];

  const allEvents = [...incidentEvents, ...coordinationEvents].sort((a, b) =>
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const getIcon = (type: string, message: string) => {
    if (type === "coordination") return <Zap className="h-4 w-4 text-yellow-500" />;
    if (message.toLowerCase().includes("block")) return <Shield className="h-4 w-4 text-red-500" />;
    if (message.toLowerCase().includes("scan")) return <Search className="h-4 w-4 text-blue-500" />;
    return <Activity className="h-4 w-4 text-muted-foreground" />;
  };

  return (
    <div className="grid gap-6 md:grid-cols-3">
      <div className="md:col-span-2">
        <Card className="h-full">
            <CardHeader>
                <CardTitle>Unified Timeline</CardTitle>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-[600px] pr-4">
                    <div className="space-y-8 relative pl-6 border-l-2 border-muted">
                        {allEvents.map((event, idx) => (
                            <div key={idx} className="relative">
                                <div className="absolute -left-[31px] top-1 h-6 w-6 rounded-full bg-background border flex items-center justify-center z-10">
                                    {getIcon(event.type, event.message)}
                                </div>
                                <div className="flex flex-col gap-1">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-mono text-muted-foreground" suppressHydrationWarning>
                                            {new Date(event.timestamp).toLocaleString()}
                                        </span>
                                        <Badge variant={event.type === "coordination" ? "secondary" : "outline"}>
                                            {event.type === "coordination" ? "System" : "Network"}
                                        </Badge>
                                    </div>
                                    <p className="text-sm font-medium">{event.message}</p>
                                    <p className="text-xs text-muted-foreground">Source: {event.source}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
      </div>

      <div>
         <Card>
            <CardHeader>
                <CardTitle className="text-sm">Event Distribution</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    <div className="flex justify-between text-sm">
                        <span>Total Events</span>
                        <span className="font-bold">{incident.detailed_events?.length || 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                        <span>Coordination Actions</span>
                        <span className="font-bold">{coordination?.coordination_timeline?.length || 0}</span>
                    </div>
                    <Separator />
                    <div className="text-xs text-muted-foreground" suppressHydrationWarning>
                        First Seen: {incidentEvents[incidentEvents.length - 1]?.timestamp ? new Date(incidentEvents[incidentEvents.length - 1].timestamp).toLocaleString() : "N/A"}
                    </div>
                     <div className="text-xs text-muted-foreground" suppressHydrationWarning>
                        Last Seen: {incidentEvents[0]?.timestamp ? new Date(incidentEvents[0].timestamp).toLocaleString() : "N/A"}
                    </div>
                </div>
            </CardContent>
         </Card>
      </div>
    </div>
  );
}
