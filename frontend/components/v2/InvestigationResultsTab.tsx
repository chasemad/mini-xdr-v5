"use client";

import React, { useState, useEffect } from "react";
import {
    Search, ChevronRight, AlertTriangle, CheckCircle,
    Clock, Loader2, Download, Terminal, AlertCircle,
    XCircle, Bot, Filter
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { apiUrl } from "@/app/utils/api";

interface Investigation {
    id: number;
    investigation_id: string;
    tool_name: string;
    tool_category: string;
    status: "running" | "completed" | "failed";
    started_at: string;
    completed_at?: string;
    execution_time_ms?: number;
    parameters: Record<string, any>;
    results?: {
        summary: string;
        findings: Array<{
            title: string;
            description: string;
            severity: "critical" | "high" | "medium" | "low";
            iocs?: string[];
        }>;
        recommendations?: string[];
        evidence?: any;
    };
    findings_count: number;
    iocs_discovered?: string[];
    severity?: string;
    confidence_score: number;
    triggered_by: string;
    error_message?: string;
    exported: boolean;
}

interface InvestigationResultsTabProps {
    incident: any;
    onExecuteTool: (toolName: string, params: any) => void;
}

export default function InvestigationResultsTab({
    incident,
    onExecuteTool,
}: InvestigationResultsTabProps) {
    const [investigations, setInvestigations] = useState<Investigation[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState<string>("all");
    const [selectedInvestigation, setSelectedInvestigation] = useState<Investigation | null>(null);
    const [exporting, setExporting] = useState<string | null>(null);

    // Fetch investigations
    const fetchInvestigations = async () => {
        try {
            const response = await fetch(
                apiUrl(`/api/incidents/${incident.id}/investigations`)
            );
            const data = await response.json();
            setInvestigations(data);
        } catch (error) {
            console.error("Failed to fetch investigations:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchInvestigations();
        const interval = setInterval(fetchInvestigations, 5000); // Poll every 5 seconds
        return () => clearInterval(interval);
    }, [incident.id]);

    // Export investigation
    const handleExport = async (investigationId: string, format: string) => {
        setExporting(investigationId);
        try {
            const response = await fetch(
                apiUrl(`/api/investigations/${investigationId}/export`),
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ format }),
                }
            );
            const data = await response.json();

            if (format === "markdown") {
                // Download markdown file
                const blob = new Blob([data.content], { type: "text/markdown" });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `investigation_${investigationId}.md`;
                a.click();
            } else if (format === "json") {
                const blob = new Blob([JSON.stringify(data, null, 2)], {
                    type: "application/json",
                });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `investigation_${investigationId}.json`;
                a.click();
            }
        } catch (error) {
            console.error("Export failed:", error);
        } finally {
            setExporting(null);
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case "completed":
                return <CheckCircle className="w-4 h-4 text-green-500" />;
            case "failed":
                return <XCircle className="w-4 h-4 text-red-500" />;
            case "running":
                return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
            default:
                return <Clock className="w-4 h-4 text-gray-500" />;
        }
    };

    const getSeverityColor = (severity?: string) => {
        switch (severity?.toLowerCase()) {
            case "critical":
                return "text-red-600 bg-red-100 dark:bg-red-900/30 dark:text-red-300";
            case "high":
                return "text-orange-600 bg-orange-100 dark:bg-orange-900/30 dark:text-orange-300";
            case "medium":
                return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300";
            case "low":
                return "text-blue-600 bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300";
            default:
                return "text-gray-600 bg-gray-100 dark:bg-gray-900/30 dark:text-gray-300";
        }
    };

    const filteredInvestigations = investigations.filter((inv) => {
        // Search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            if (
                !inv.tool_name.toLowerCase().includes(query) &&
                !inv.tool_category.toLowerCase().includes(query)
            ) {
                return false;
            }
        }

        // Status filter
        if (statusFilter !== "all" && inv.status !== statusFilter) {
            return false;
        }

        return true;
    });

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col">
            {/* Toolbar */}
            <div className="flex items-center justify-between px-4 py-3 border-b bg-background/50 backdrop-blur-sm shrink-0">
                <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-primary" />
                    <h3 className="text-sm font-semibold">Investigation Results</h3>
                    <Badge variant="secondary" className="text-[9px] h-4 px-1.5">
                        {investigations.length}
                    </Badge>
                </div>

                <div className="flex items-center gap-2">
                    {/* Status Filter */}
                    <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value)}
                        className="h-7 text-xs border rounded px-2 bg-background"
                    >
                        <option value="all">All Status</option>
                        <option value="running">Running</option>
                        <option value="completed">Completed</option>
                        <option value="failed">Failed</option>
                    </select>

                    {/* Search */}
                    <div className="relative w-48">
                        <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground" />
                        <Input
                            className="h-7 pl-7 pr-3 text-xs"
                            placeholder="Search investigations..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                </div>
            </div>

            <div className="flex-1 flex min-h-0">
                {/* Investigation List */}
                <div className={cn("border-r", selectedInvestigation ? "w-1/3" : "w-full")}>
                    <ScrollArea className="h-full">
                        <div className="divide-y">
                            {filteredInvestigations.map((inv) => (
                                <button
                                    key={inv.id}
                                    onClick={() => setSelectedInvestigation(inv)}
                                    className={cn(
                                        "w-full p-4 text-left hover:bg-muted/50 transition-colors",
                                        selectedInvestigation?.id === inv.id && "bg-muted/50"
                                    )}
                                >
                                    <div className="flex items-start gap-3">
                                        <div className="mt-0.5">{getStatusIcon(inv.status)}</div>

                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-medium">
                                                    {inv.tool_name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                                                </h4>
                                                {inv.severity && (
                                                    <Badge
                                                        variant="outline"
                                                        className={cn("text-[9px] h-4 px-1", getSeverityColor(inv.severity))}
                                                    >
                                                        {inv.severity.toUpperCase()}
                                                    </Badge>
                                                )}
                                            </div>

                                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                                                <Badge variant="outline" className="text-[8px] h-3.5 px-1">
                                                    {inv.tool_category}
                                                </Badge>
                                                <span>•</span>
                                                <span suppressHydrationWarning>
                                                    {new Date(inv.started_at).toLocaleString()}
                                                </span>
                                                {inv.findings_count > 0 && (
                                                    <>
                                                        <span>•</span>
                                                        <span className="text-primary font-medium">
                                                            {inv.findings_count} finding{inv.findings_count !== 1 ? "s" : ""}
                                                        </span>
                                                    </>
                                                )}
                                            </div>

                                            {inv.execution_time_ms && inv.status === "completed" && (
                                                <div className="text-[9px] text-muted-foreground mt-1">
                                                    Completed in {inv.execution_time_ms}ms
                                                </div>
                                            )}

                                            {inv.error_message && (
                                                <div className="text-[10px] text-red-600 dark:text-red-400 mt-1 flex items-center gap-1">
                                                    <AlertCircle className="w-3 h-3" />
                                                    {inv.error_message}
                                                </div>
                                            )}
                                        </div>

                                        <ChevronRight className="w-4 h-4 shrink-0 text-muted-foreground" />
                                    </div>
                                </button>
                            ))}

                            {filteredInvestigations.length === 0 && (
                                <div className="p-8 text-center">
                                    <Bot className="w-12 h-12 mx-auto mb-3 opacity-20" />
                                    <p className="text-sm text-muted-foreground">
                                        {searchQuery || statusFilter !== "all"
                                            ? "No investigations match your filters"
                                            : "No investigations run yet"}
                                    </p>
                                    {!searchQuery && statusFilter === "all" && (
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            className="mt-3"
                                            onClick={() => onExecuteTool("threat_hunting", {})}
                                        >
                                            Run First Investigation
                                        </Button>
                                    )}
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                </div>

                {/* Investigation Details */}
                {selectedInvestigation && (
                    <ScrollArea className="flex-1">
                        <div className="p-4">
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div>
                                    <h2 className="text-lg font-semibold mb-1">
                                        {selectedInvestigation.tool_name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                                    </h2>
                                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                        <Badge variant="outline" className="text-[9px]">
                                            {selectedInvestigation.tool_category}
                                        </Badge>
                                        <span>•</span>
                                        <span>ID: {selectedInvestigation.investigation_id}</span>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2">
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        onClick={() => handleExport(selectedInvestigation.investigation_id, "json")}
                                        disabled={exporting === selectedInvestigation.investigation_id}
                                    >
                                        {exporting === selectedInvestigation.investigation_id ? (
                                            <Loader2 className="w-3 h-3 animate-spin mr-1" />
                                        ) : (
                                            <Download className="w-3 h-3 mr-1" />
                                        )}
                                        Export JSON
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        onClick={() => handleExport(selectedInvestigation.investigation_id, "markdown")}
                                        disabled={exporting === selectedInvestigation.investigation_id}
                                    >
                                        <Download className="w-3 h-3 mr-1" />
                                        Export MD
                                    </Button>
                                </div>
                            </div>

                            {/* Summary */}
                            {selectedInvestigation.results?.summary && (
                                <div className="bg-muted/30 rounded-lg p-4 mb-4">
                                    <h3 className="text-xs font-semibold mb-2 text-muted-foreground uppercase">Summary</h3>
                                    <p className="text-sm">{selectedInvestigation.results.summary}</p>
                                </div>
                            )}

                            {/* Findings */}
                            {selectedInvestigation.results?.findings && selectedInvestigation.results.findings.length > 0 && (
                                <div className="mb-4">
                                    <h3 className="text-xs font-semibold mb-3 text-muted-foreground uppercase flex items-center gap-2">
                                        <AlertTriangle className="w-3.5 h-3.5" />
                                        Findings ({selectedInvestigation.results.findings.length})
                                    </h3>
                                    <div className="space-y-3">
                                        {selectedInvestigation.results.findings.map((finding, idx) => (
                                            <div
                                                key={idx}
                                                className={cn(
                                                    "border rounded-lg p-3",
                                                    getSeverityColor(finding.severity)
                                                )}
                                            >
                                                <div className="flex items-start justify-between mb-2">
                                                    <h4 className="text-sm font-semibold">{finding.title}</h4>
                                                    <Badge
                                                        variant="outline"
                                                        className={cn("text-[8px]", getSeverityColor(finding.severity))}
                                                    >
                                                        {finding.severity.toUpperCase()}
                                                    </Badge>
                                                </div>
                                                <p className="text-xs text-foreground/80">{finding.description}</p>
                                                {finding.iocs && finding.iocs.length > 0 && (
                                                    <div className="mt-2">
                                                        <span className="text-[10px] font-medium">IOCs:</span>
                                                        <div className="flex flex-wrap gap-1 mt-1">
                                                            {finding.iocs.map((ioc, i) => (
                                                                <Badge key={i} variant="secondary" className="text-[8px] font-mono">
                                                                    {ioc}
                                                                </Badge>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Recommendations */}
                            {selectedInvestigation.results?.recommendations && selectedInvestigation.results.recommendations.length > 0 && (
                                <div className="mb-4">
                                    <h3 className="text-xs font-semibold mb-2 text-muted-foreground uppercase">Recommendations</h3>
                                    <ul className="space-y-1.5">
                                        {selectedInvestigation.results.recommendations.map((rec, idx) => (
                                            <li key={idx} className="text-sm flex items-start gap-2">
                                                <ChevronRight className="w-3.5 h-3.5 mt-0.5 shrink-0 text-primary" />
                                                <span>{rec}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Metadata */}
                            <div className="border-t pt-4 mt-4">
                                <h3 className="text-xs font-semibold mb-3 text-muted-foreground uppercase">Execution Details</h3>
                                <div className="grid grid-cols-2 gap-3 text-xs">
                                    <div>
                                        <span className="text-muted-foreground">Status:</span>
                                        <div className="flex items-center gap-2 mt-1">
                                            {getStatusIcon(selectedInvestigation.status)}
                                            <span className="font-medium">{selectedInvestigation.status}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <span className="text-muted-foreground">Triggered By:</span>
                                        <div className="mt-1 font-medium">{selectedInvestigation.triggered_by}</div>
                                    </div>
                                    <div>
                                        <span className="text-muted-foreground">Started:</span>
                                        <div className="mt-1 font-mono text-[10px]" suppressHydrationWarning>
                                            {new Date(selectedInvestigation.started_at).toLocaleString()}
                                        </div>
                                    </div>
                                    {selectedInvestigation.completed_at && (
                                        <div>
                                            <span className="text-muted-foreground">Completed:</span>
                                            <div className="mt-1 font-mono text-[10px]" suppressHydrationWarning>
                                                {new Date(selectedInvestigation.completed_at).toLocaleString()}
                                            </div>
                                        </div>
                                    )}
                                    {selectedInvestigation.confidence_score > 0 && (
                                        <div>
                                            <span className="text-muted-foreground">Confidence:</span>
                                            <div className="mt-1 font-medium">
                                                {(selectedInvestigation.confidence_score * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </ScrollArea>
                )}
            </div>
        </div>
    );
}
