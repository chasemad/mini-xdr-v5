"use client";

import React, { useState, useMemo } from "react";
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
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
    Activity, Search, Terminal, Clock,
    AlertTriangle, ChevronRight, BarChart2, List,
    Filter
} from "lucide-react";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuCheckboxItem,
    DropdownMenuTrigger,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import InteractiveTimeline from "./InteractiveTimeline";
import { getEventPriority, getSeverityLevel, getSeverityColor, getSeverityBg } from "@/lib/event-utils";

interface EventsHistorySheetProps {
    isOpen: boolean;
    onClose: () => void;
    events: any[];
    incidentId?: string;
}

export default function EventsHistorySheet({ isOpen, onClose, events, incidentId }: EventsHistorySheetProps) {
    const [selectedEvent, setSelectedEvent] = useState<any>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [viewMode, setViewMode] = useState<"list" | "graph">("list");
    const [severityFilter, setSeverityFilter] = useState<string[]>([]);
    const [timeFilter, setTimeFilter] = useState<string>("all");

    // Filter and sort events
    const filteredEvents = useMemo(() => {
        let result = [...events];

        // Sort by timestamp descending
        result.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());

        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            result = result.filter(event =>
                (event.message?.toLowerCase().includes(query)) ||
                (event.eventid?.toLowerCase().includes(query)) ||
                (event.source?.toLowerCase().includes(query))
            );
        }

        if (severityFilter.length > 0) {
            result = result.filter(event => {
                const priority = getEventPriority(event);
                const severity = getSeverityLevel(priority);
                return severityFilter.includes(severity);
            });
        }

        if (timeFilter !== "all") {
            const now = Date.now();
            const hours = timeFilter === "1h" ? 1 : 24;
            result = result.filter(event => {
                const eventTime = new Date(event.ts).getTime();
                return now - eventTime <= hours * 60 * 60 * 1000;
            });
        }

        return result;
    }, [events, searchQuery, severityFilter, timeFilter]);

    // Group events for graph view if needed, but InteractiveTimeline handles raw events too
    // We'll pass filtered events to the timeline so the graph reflects the search

    return (
        <Sheet open={isOpen} onOpenChange={onClose}>
            <SheetContent className="w-[900px] sm:max-w-[900px] p-0 flex flex-col bg-background border-l">
                <div className="flex flex-1 overflow-hidden">

                    {/* Left Side: Event List / Graph */}
                    <div className="w-[40%] border-r flex flex-col bg-muted/5">
                        <SheetHeader className="p-4 border-b shrink-0 space-y-4">
                            <div className="flex items-center justify-between">
                                <SheetTitle className="text-base flex items-center gap-2">
                                    <Activity className="w-4 h-4 text-primary" />
                                    Event History
                                </SheetTitle>
                                <Badge variant="outline" className="font-mono">
                                    {filteredEvents.length} events
                                </Badge>
                            </div>

                            <div className="flex flex-col gap-2">
                                <div className="flex gap-2">
                                    <div className="relative flex-1">
                                        <Search className="absolute left-2 top-2 w-3.5 h-3.5 text-muted-foreground" />
                                        <Input
                                            placeholder="Filter events..."
                                            className="h-8 pl-8 text-xs bg-background"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                        />
                                    </div>
                                    <div className="flex bg-muted rounded-md p-0.5 shrink-0">
                                        <Button
                                            variant={viewMode === 'list' ? 'secondary' : 'ghost'}
                                            size="icon"
                                            className="h-7 w-7"
                                            onClick={() => setViewMode('list')}
                                            title="List View"
                                        >
                                            <List className="w-3.5 h-3.5" />
                                        </Button>
                                        <Button
                                            variant={viewMode === 'graph' ? 'secondary' : 'ghost'}
                                            size="icon"
                                            className="h-7 w-7"
                                            onClick={() => setViewMode('graph')}
                                            title="Graph View"
                                        >
                                            <BarChart2 className="w-3.5 h-3.5" />
                                        </Button>
                                    </div>
                                </div>
                                <div className="flex gap-2">
                                    <DropdownMenu>
                                        <DropdownMenuTrigger asChild>
                                            <Button variant="outline" size="sm" className={cn("h-7 text-xs flex-1", severityFilter.length > 0 && "border-primary text-primary bg-primary/5")}>
                                                <Filter className="w-3 h-3 mr-1" />
                                                {severityFilter.length > 0 ? `${severityFilter.length} Severity` : "Severity"}
                                            </Button>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent align="start">
                                            <DropdownMenuLabel>Filter Severity</DropdownMenuLabel>
                                            <DropdownMenuSeparator />
                                            {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map((level) => (
                                                <DropdownMenuCheckboxItem
                                                    key={level}
                                                    checked={severityFilter.includes(level)}
                                                    onCheckedChange={(checked) => {
                                                        setSeverityFilter(prev =>
                                                            checked ? [...prev, level] : prev.filter(l => l !== level)
                                                        );
                                                    }}
                                                >
                                                    {level}
                                                </DropdownMenuCheckboxItem>
                                            ))}
                                        </DropdownMenuContent>
                                    </DropdownMenu>

                                    <DropdownMenu>
                                        <DropdownMenuTrigger asChild>
                                            <Button variant="outline" size="sm" className={cn("h-7 text-xs flex-1", timeFilter !== "all" && "border-primary text-primary bg-primary/5")}>
                                                <Clock className="w-3 h-3 mr-1" />
                                                {timeFilter === "all" ? "All Time" : timeFilter}
                                            </Button>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent align="start">
                                            <DropdownMenuLabel>Filter Time</DropdownMenuLabel>
                                            <DropdownMenuSeparator />
                                            <DropdownMenuRadioGroup value={timeFilter} onValueChange={setTimeFilter}>
                                                <DropdownMenuRadioItem value="all">All Time</DropdownMenuRadioItem>
                                                <DropdownMenuRadioItem value="1h">Last Hour</DropdownMenuRadioItem>
                                                <DropdownMenuRadioItem value="24h">Last 24 Hours</DropdownMenuRadioItem>
                                            </DropdownMenuRadioGroup>
                                        </DropdownMenuContent>
                                    </DropdownMenu>
                                </div>
                            </div>
                        </SheetHeader>

                        <div className="flex-1 min-h-0 flex flex-col">
                            {viewMode === 'list' ? (
                                <ScrollArea className="flex-1">
                                    <div className="divide-y divide-border/40">
                                        {filteredEvents.map((event, idx) => {
                                            const priority = getEventPriority(event);
                                            const severityColor = getSeverityColor(priority);

                                            return (
                                            <button
                                                key={idx}
                                                onClick={() => setSelectedEvent(event)}
                                                className={cn(
                                                    "w-full text-left p-3 hover:bg-muted/50 transition-colors focus:outline-none flex flex-col gap-1.5 group border-l-2",
                                                    selectedEvent === event ? "bg-accent border-primary" : "hover:border-primary/30 border-transparent",
                                                    priority >= 90 && !selectedEvent && "bg-red-50/50 dark:bg-red-950/20",
                                                    priority >= 70 && priority < 90 && !selectedEvent && "bg-orange-50/50 dark:bg-orange-950/20"
                                                )}
                                            >
                                                <div className="flex items-start justify-between w-full gap-2">
                                                    <span className="font-mono text-[10px] text-muted-foreground shrink-0 mt-0.5">
                                                        {new Date(event.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                                    </span>
                                                    <Badge variant="outline" className={cn("text-[9px] h-4 px-1 font-normal opacity-70 shrink-0", severityColor)}>
                                                        {event.eventid}
                                                    </Badge>
                                                </div>
                                                <div className="text-xs text-foreground/90 line-clamp-2 break-words leading-relaxed">
                                                    {event.message}
                                                </div>
                                            </button>
                                        )})}
                                        {filteredEvents.length === 0 && (
                                            <div className="p-8 text-center text-muted-foreground text-xs">
                                                No events match your filter.
                                            </div>
                                        )}
                                    </div>
                                </ScrollArea>
                            ) : (
                                <div className="flex-1 p-4 flex flex-col">
                                    <div className="text-xs font-semibold mb-4 text-muted-foreground">Event Frequency</div>
                                    <div className="flex-1 min-h-0">
                                        <InteractiveTimeline events={filteredEvents} />
                                    </div>
                                    <div className="mt-4 text-[10px] text-muted-foreground text-center">
                                        Showing distribution of {filteredEvents.length} filtered events
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right Side: Event Details */}
                    <div className="flex-1 flex flex-col bg-background">
                        {selectedEvent ? (
                            <div className="flex-1 flex flex-col overflow-hidden">
                                <div className="p-6 pb-4 border-b bg-card shrink-0">
                                    <div className="flex items-start gap-3 mb-2">
                                        <div className="p-2 rounded-lg border bg-primary/5 border-primary/10 mt-1">
                                            <Terminal className="w-4 h-4 text-primary" />
                                        </div>
                                        <div className="flex-1">
                                            <h2 className="text-sm font-bold leading-relaxed font-mono text-primary">
                                                {selectedEvent.eventid}
                                            </h2>
                                            <div className="text-xs text-muted-foreground mt-1">
                                                <span suppressHydrationWarning>{new Date(selectedEvent.ts).toLocaleString()}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="mt-2 p-3 bg-muted/30 rounded-md border text-sm leading-relaxed">
                                        {selectedEvent.message}
                                    </div>
                                </div>

                                <ScrollArea className="flex-1">
                                    <div className="p-6 space-y-6">

                                        {/* Raw Data */}
                                        <div className="space-y-2">
                                            <h3 className="text-sm font-semibold flex items-center gap-2">
                                                <Activity className="w-4 h-4 text-muted-foreground" />
                                                Event Data
                                            </h3>
                                            <div className="relative group">
                                                <pre className="p-4 rounded-md bg-zinc-950 text-zinc-50 text-xs overflow-x-auto font-mono border border-zinc-800 leading-normal">
                                                    {JSON.stringify(selectedEvent, null, 2)}
                                                </pre>
                                            </div>
                                        </div>

                                    </div>
                                </ScrollArea>
                            </div>
                        ) : (
                            <div className="flex-1 flex items-center justify-center flex-col text-muted-foreground gap-3">
                                <Activity className="w-12 h-12 opacity-20" />
                                <p className="text-sm">Select an event to view full details</p>
                            </div>
                        )}
                    </div>
                </div>
            </SheetContent>
        </Sheet>
    );
}
