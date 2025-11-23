"use client";

import React, { useState } from "react";
import {
    Bot, CheckCircle, Search, AlertTriangle,
    Activity, Clock, Terminal, ChevronRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface InvestigationWorkspaceProps {
    activeIncident: any;
    allActions: any[];
    onActionClick: (action: any) => void;
    onHistoryOpen: () => void;
}

export default function InvestigationWorkspace({
    activeIncident,
    allActions,
    onActionClick,
    onHistoryOpen
}: InvestigationWorkspaceProps) {
    const [searchQuery, setSearchQuery] = useState("");

    // Simple timeline visualization using CSS
    const renderEventTimeline = () => {
        const events = activeIncident.detailed_events || [];
        if (events.length === 0) {
            return (
                <div className="h-full flex items-center justify-center text-muted-foreground text-xs">
                    No timeline data available
                </div>
            );
        }

        // Group events by minute
        const eventCounts: Record<string, number> = {};
        events.forEach((event: any) => {
            const time = new Date(event.ts);
            const key = `${time.getHours()}:${String(time.getMinutes()).padStart(2, '0')}`;
            eventCounts[key] = (eventCounts[key] || 0) + 1;
        });

        const maxCount = Math.max(...Object.values(eventCounts));
        const entries = Object.entries(eventCounts).slice(-20); // Last 20 time buckets

        return (
            <div className="h-full flex items-end gap-1 px-2">
                {entries.map(([time, count], idx) => {
                    const height = (count / maxCount) * 100;
                    return (
                        <div key={idx} className="flex-1 flex flex-col items-center gap-1 group">
                            <div
                                className="w-full bg-primary/70 hover:bg-primary transition-colors rounded-t-sm cursor-pointer"
                                style={{ height: `${height}%`, minHeight: '8px' }}
                                title={`${time}: ${count} events`}
                            />
                            {idx % 3 === 0 && (
                                <span className="text-[9px] text-muted-foreground font-mono">{time}</span>
                            )}
                        </div>
                    );
                })}
            </div>
        );
    };

    const filteredEvents = (activeIncident.detailed_events || []).filter((event: any) => {
        if (!searchQuery) return true;
        const query = searchQuery.toLowerCase();
        return (
            event.message?.toLowerCase().includes(query) ||
            event.eventid?.toLowerCase().includes(query)
        );
    });

    return (
        <section className="h-full flex flex-col overflow-hidden bg-background relative">

            {/* Event Timeline - Simplified */}
            <div className="h-32 border-b bg-gradient-to-b from-muted/5 to-muted/10 p-4 flex flex-col shrink-0">
                <div className="flex justify-between items-center mb-3 shrink-0">
                    <h3 className="text-xs font-semibold text-foreground flex items-center gap-2">
                        <Activity className="w-3.5 h-3.5 text-primary" />
                        Event Timeline
                    </h3>
                    <div className="flex gap-1">
                        <Badge variant="outline" className="text-[9px] h-5 px-1.5 cursor-pointer hover:bg-muted">1h</Badge>
                        <Badge variant="outline" className="text-[9px] h-5 px-1.5 cursor-pointer hover:bg-muted">24h</Badge>
                    </div>
                </div>
                <div className="flex-1 min-h-0">
                    {renderEventTimeline()}
                </div>
            </div>

            {/* Workspace Content */}
            <div className="flex-1 flex flex-col min-h-0">

                {/* Events Area - Improved */}
                <div className="flex-1 flex flex-col min-h-[200px] bg-background">
                    <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/10 backdrop-blur-sm shrink-0">
                        <div className="flex gap-1">
                            <Button variant="ghost" size="sm" className="h-6 text-[10px] font-medium bg-primary/10 text-primary">
                                Events
                            </Button>
                            <Button variant="ghost" size="sm" className="h-6 text-[10px] font-medium text-muted-foreground">
                                Graph
                            </Button>
                        </div>
                        <div className="relative w-48">
                            <Search className="absolute left-2 top-1.5 w-3 h-3 text-muted-foreground" />
                            <Input
                                className="w-full h-6 pl-7 pr-3 text-[10px] bg-background border-muted focus-visible:ring-1"
                                placeholder="Filter events..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                    </div>

                    <ScrollArea className="flex-1">
                        <div className="p-4 space-y-3">

                            {/* Triggering Events Context */}
                            {activeIncident.triggering_events && activeIncident.triggering_events.length > 0 && (
                                <div className="bg-orange-50/50 dark:bg-orange-950/20 rounded-lg border border-orange-200/50 dark:border-orange-800/50 p-3">
                                    <div className="flex items-center gap-2 mb-2">
                                        <AlertTriangle className="w-3.5 h-3.5 text-orange-500" />
                                        <span className="text-xs font-semibold text-orange-800 dark:text-orange-200">
                                            Incident Trigger Context
                                        </span>
                                        <Badge variant="outline" className="text-[9px] h-4 px-1 bg-orange-100 dark:bg-orange-900/30 border-orange-300 dark:border-orange-700">
                                            {activeIncident.triggering_events.length} events
                                        </Badge>
                                    </div>
                                    <div className="space-y-1.5 mt-2">
                                        {activeIncident.triggering_events.map((event: any, idx: number) => (
                                            <div
                                                key={event.id || idx}
                                                className={cn(
                                                    "flex items-start gap-2 p-2 rounded-md text-[10px] border",
                                                    event.is_trigger
                                                        ? "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800"
                                                        : "bg-background/50 border-border/50"
                                                )}
                                            >
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        {event.is_trigger && (
                                                            <span className="text-red-600 dark:text-red-400 font-bold text-[9px]">⚠️ TRIGGER</span>
                                                        )}
                                                        <span className="text-muted-foreground font-mono text-[9px]">
                                                            {new Date(event.ts).toLocaleTimeString()}
                                                        </span>
                                                        <Badge variant="outline" className="text-[8px] h-3.5 px-1">
                                                            {event.eventid}
                                                        </Badge>
                                                    </div>
                                                    <div className="text-foreground/90 leading-relaxed break-words">
                                                        {event.message}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Event Summary */}
                            {activeIncident.event_summary && (
                                <div className="bg-muted/30 rounded-lg border border-border/50 p-3">
                                    <div className="text-xs font-semibold text-foreground mb-2 flex items-center gap-2">
                                        <Terminal className="w-3.5 h-3.5 text-primary" />
                                        Complete Event History
                                    </div>
                                    <div className="grid grid-cols-2 gap-3 text-[10px]">
                                        <div className="flex items-center gap-2">
                                            <span className="text-muted-foreground">Total Events:</span>
                                            <span className="font-mono font-bold text-foreground">
                                                {activeIncident.event_summary.total_events}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-muted-foreground">Event Types:</span>
                                            <span className="font-mono text-foreground">
                                                {activeIncident.event_summary.event_types?.join(', ') || 'None'}
                                            </span>
                                        </div>
                                    </div>
                                    {activeIncident.event_summary.event_counts_by_type &&
                                        Object.keys(activeIncident.event_summary.event_counts_by_type).length > 0 && (
                                            <div className="mt-2">
                                                <div className="flex flex-wrap gap-1">
                                                    {Object.entries(activeIncident.event_summary.event_counts_by_type).map(([type, count]: [string, any]) => (
                                                        <Badge key={type} variant="outline" className="text-[9px] h-4 px-1.5 bg-background">
                                                            {type}: {count}
                                                        </Badge>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                </div>
                            )}

                            {/* Individual Events - Improved Layout */}
                            <div className="space-y-0.5">
                                {filteredEvents.length > 0 ? (
                                    filteredEvents.map((event: any, idx: number) => (
                                        <div
                                            key={idx}
                                            className="flex gap-3 p-2 px-3 hover:bg-muted/40 rounded-md text-[10px] group transition-all border border-transparent hover:border-border/50"
                                        >
                                            <div className="flex items-center gap-2 shrink-0">
                                                <Clock className="w-3 h-3 text-muted-foreground/50 group-hover:text-muted-foreground" />
                                                <span className="text-muted-foreground font-mono tabular-nums w-16">
                                                    {new Date(event.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                                </span>
                                            </div>
                                            <div className="flex-1 min-w-0 font-mono">
                                                <Badge variant="outline" className="text-[8px] h-3.5 px-1 mr-2 bg-primary/5 border-primary/20">
                                                    {event.eventid}
                                                </Badge>
                                                <span className="text-foreground/90 break-words">{event.message}</span>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="p-8 text-center">
                                        <Terminal className="w-8 h-8 mx-auto mb-2 opacity-20" />
                                        <p className="text-xs text-muted-foreground">
                                            {searchQuery ? 'No events match your filter' : 'No events found'}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </ScrollArea>
                </div>

                {/* AI Agent Actions - Improved */}
                <div className="flex-1 border-b flex flex-col min-h-[180px] bg-card/30">
                    <div className="flex items-center justify-between px-4 py-2.5 border-b bg-background/80 backdrop-blur-sm shrink-0">
                        <h3 className="text-xs font-semibold flex items-center gap-2 text-foreground">
                            <Bot className="w-4 h-4 text-primary" />
                            AI Agent Actions
                            <Badge variant="secondary" className="text-[9px] h-4 px-1.5 ml-1">
                                {allActions.length}
                            </Badge>
                        </h3>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 text-[9px] text-muted-foreground hover:text-primary gap-1"
                            onClick={onHistoryOpen}
                        >
                            View All
                            <ChevronRight className="w-3 h-3" />
                        </Button>
                    </div>

                    <ScrollArea className="flex-1">
                        <div className="divide-y divide-border/20">
                            {allActions.slice(0, 10).map((action: any, idx: number) => (
                                <button
                                    key={`${action.id ?? action.action ?? 'action'}-${idx}`}
                                    onClick={() => onActionClick(action)}
                                    className={cn(
                                        "w-full flex items-start gap-3 px-4 py-2.5",
                                        "hover:bg-primary/5 transition-all group text-left",
                                        "border-l-2 border-transparent hover:border-primary/50"
                                    )}
                                >
                                    <div className="mt-1">
                                        <div className="w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center">
                                            <CheckCircle className="w-3 h-3 text-green-500" />
                                        </div>
                                    </div>

                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium text-xs group-hover:text-primary transition-colors">
                                            {action.action}
                                        </div>
                                        {action.detail && (
                                            <div className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">
                                                {action.detail}
                                            </div>
                                        )}
                                        {action.agent_type && (
                                            <Badge variant="outline" className="text-[8px] h-3.5 px-1 mt-1.5 border-primary/20">
                                                {action.agent_type === 'ai' ? '🤖 AI' : action.agent_type === 'playbook' ? '📋 Playbook' : action.agent_type}
                                            </Badge>
                                        )}
                                    </div>

                                    <div className="flex flex-col items-end gap-1 shrink-0">
                                        <span className="text-[9px] text-muted-foreground font-mono tabular-nums">
                                            {new Date(action.timestamp || action.created_at || Date.now()).toLocaleTimeString([], {
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </span>
                                    </div>
                                </button>
                            ))}

                            {allActions.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-32 text-muted-foreground gap-2">
                                    <Bot className="w-8 h-8 opacity-20" />
                                    <span className="text-xs">No agent actions recorded yet.</span>
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                </div>
            </div>
        </section>
    );
}
