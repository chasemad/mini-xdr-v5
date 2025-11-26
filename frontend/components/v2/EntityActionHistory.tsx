import React, { useState } from "react";
import {
    Sheet,
    SheetContent,
    SheetHeader,
    SheetTitle,
    SheetDescription,
} from "@/components/ui/sheet";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Shield, Activity, User, Clock, CheckCircle, XCircle, Terminal, AlertCircle, RotateCcw, Filter, Bot, UserCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Action } from "@/app/types";

interface EntityActionHistoryProps {
    isOpen: boolean;
    onClose: () => void;
    entityId: string;
    entityType?: string;
    actions: Partial<Action>[];
    onRollback?: (actionId: string) => void;
}

export default function EntityActionHistory({
    isOpen,
    onClose,
    entityId,
    entityType = "Entity",
    actions,
    onRollback
}: EntityActionHistoryProps) {
    const [selectedAction, setSelectedAction] = useState<Partial<Action> | null>(null);
    const [filter, setFilter] = useState<'all' | 'agent' | 'manual'>('all');
    const [rollbackConfirmOpen, setRollbackConfirmOpen] = useState(false);
    const [actionToRollback, setActionToRollback] = useState<Partial<Action> | null>(null);

    // Filter actions relevant to this entity
    const entityActions = actions.filter(a => {
        // 1. Entity Match
        let match = false;
        if (a.scope?.type === 'entity' && a.scope?.target === entityId) match = true;
        else if (a.parameters && JSON.stringify(a.parameters).includes(entityId)) match = true;

        if (!match) return false;

        // 2. Type Filter
        if (filter === 'agent') return !!(a.agent || a.agent_type);
        if (filter === 'manual') return !(a.agent || a.agent_type);

        return true;
    });

    const sortedActions = [...entityActions].sort((a, b) => {
        const dateA = new Date(a.timestamp || a.created_at || 0).getTime();
        const dateB = new Date(b.timestamp || b.created_at || 0).getTime();
        return dateB - dateA;
    });

    const handleRollbackClick = (action: Partial<Action>) => {
        setActionToRollback(action);
        setRollbackConfirmOpen(true);
    };

    const confirmRollback = () => {
        if (actionToRollback && actionToRollback.rollback_id && onRollback) {
            onRollback(actionToRollback.rollback_id);
        }
        setRollbackConfirmOpen(false);
        setActionToRollback(null);
    };

    const getStatusIcon = (status: string) => {
        switch (status?.toLowerCase()) {
            case 'completed':
            case 'success':
                return <CheckCircle className="w-4 h-4 text-green-500" />;
            case 'failed':
            case 'error':
                return <XCircle className="w-4 h-4 text-red-500" />;
            case 'pending':
            case 'running':
                return <Clock className="w-4 h-4 text-yellow-500" />;
            default:
                return <Activity className="w-4 h-4 text-muted-foreground" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status?.toLowerCase()) {
            case 'completed':
            case 'success':
                return "bg-green-500/10 text-green-500 border-green-500/20";
            case 'failed':
            case 'error':
                return "bg-red-500/10 text-red-500 border-red-500/20";
            case 'pending':
            case 'running':
                return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20";
            default:
                return "bg-muted text-muted-foreground border-border";
        }
    };

    return (
        <>
            <Sheet open={isOpen} onOpenChange={onClose}>
                <SheetContent className="w-[85vw] sm:max-w-[85vw] md:max-w-[1000px] lg:max-w-[1200px] p-0 flex flex-col bg-background border-l">
                    <div className="flex flex-1 overflow-hidden">
                        {/* Left Side: Action List */}
                        <div className="w-1/3 border-r flex flex-col bg-muted/5">
                            <SheetHeader className="p-4 border-b shrink-0 flex flex-row items-center justify-between space-y-0">
                                <div>
                                    <SheetTitle className="text-base flex items-center gap-2">
                                        <Activity className="w-4 h-4 text-blue-500" />
                                        Entity History
                                    </SheetTitle>
                                    <SheetDescription className="text-xs mt-1">
                                        {sortedActions.length} actions affecting <span className="font-mono">{entityId}</span>
                                    </SheetDescription>
                                </div>
                                <DropdownMenu>
                                    <DropdownMenuTrigger asChild>
                                        <Button variant="ghost" size="icon" className="h-8 w-8">
                                            <Filter className={cn("w-4 h-4", filter !== 'all' ? "text-primary" : "text-muted-foreground")} />
                                        </Button>
                                    </DropdownMenuTrigger>
                                    <DropdownMenuContent align="end">
                                        <DropdownMenuItem onClick={() => setFilter('all')}>
                                            All Actions
                                        </DropdownMenuItem>
                                        <DropdownMenuItem onClick={() => setFilter('agent')}>
                                            <Bot className="w-4 h-4 mr-2" />
                                            Agent Actions
                                        </DropdownMenuItem>
                                        <DropdownMenuItem onClick={() => setFilter('manual')}>
                                            <UserCircle className="w-4 h-4 mr-2" />
                                            Analyst Actions
                                        </DropdownMenuItem>
                                    </DropdownMenuContent>
                                </DropdownMenu>
                            </SheetHeader>
                            <ScrollArea className="flex-1">
                                <div className="divide-y">
                                    {sortedActions.map((action, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() => setSelectedAction(action)}
                                            className={cn(
                                                "w-full text-left p-3 hover:bg-muted/50 transition-colors focus:outline-none flex flex-col gap-2 group",
                                                selectedAction === action && "bg-accent"
                                            )}
                                        >
                                            <div className="flex items-start justify-between w-full">
                                                <span className="font-medium text-xs truncate pr-2">
                                                    {action.name || action.action || action.action_name || "Unknown Action"}
                                                </span>
                                                {getStatusIcon(action.status || 'pending')}
                                            </div>
                                            <div className="flex items-center justify-between w-full text-[10px] text-muted-foreground">
                                                <span suppressHydrationWarning>{new Date(action.timestamp || action.created_at || Date.now()).toLocaleTimeString()}</span>
                                                <Badge variant="outline" className="text-[8px] h-4 px-1 font-normal opacity-70">
                                                    {action.agent || action.agent_type ? 'AGENT' : 'ANALYST'}
                                                </Badge>
                                            </div>
                                        </button>
                                    ))}
                                    {sortedActions.length === 0 && (
                                        <div className="p-8 text-center text-muted-foreground text-xs">
                                            No actions found matching filter.
                                        </div>
                                    )}
                                </div>
                            </ScrollArea>
                        </div>

                        {/* Right Side: Action Details */}
                        <div className="flex-1 flex flex-col bg-background">
                            {selectedAction ? (
                                <div className="flex-1 flex flex-col overflow-hidden">
                                    <div className="p-6 pb-4 border-b bg-card shrink-0">
                                        <div className="flex items-center gap-3 mb-4">
                                            <div className={cn("p-2 rounded-lg border", getStatusColor(selectedAction.status || 'pending'))}>
                                                {getStatusIcon(selectedAction.status || 'pending')}
                                            </div>
                                            <div>
                                                <h2 className="text-lg font-bold leading-tight">
                                                    {selectedAction.name || selectedAction.action || selectedAction.action_name || "Unknown Action"}
                                                </h2>
                                                <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                                                    <span className="font-mono">{selectedAction.id || "ID: N/A"}</span>
                                                    <span>•</span>
                                                    <span suppressHydrationWarning>{new Date(selectedAction.timestamp || selectedAction.created_at || Date.now()).toLocaleString()}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex flex-wrap gap-2">
                                            <Badge variant="secondary" className="font-mono text-[10px] uppercase">
                                                {selectedAction.execution_method || "Automated"}
                                            </Badge>
                                            {selectedAction.agent ? (
                                                <Badge variant="outline" className="text-[10px] uppercase border-primary/30 text-primary">
                                                    {selectedAction.agent} Agent
                                                </Badge>
                                            ) : (
                                                <Badge variant="outline" className="text-[10px] uppercase border-muted-foreground/30 text-muted-foreground">
                                                    Analyst
                                                </Badge>
                                            )}
                                            {selectedAction.riskLevel && (
                                                <Badge variant="outline" className={cn("text-[10px] uppercase",
                                                    selectedAction.riskLevel === 'critical' ? "border-red-500 text-red-500" :
                                                        selectedAction.riskLevel === 'high' ? "border-orange-500 text-orange-500" :
                                                            "border-blue-500 text-blue-500"
                                                )}>
                                                    {selectedAction.riskLevel} Risk
                                                </Badge>
                                            )}
                                        </div>
                                    </div>

                                    <ScrollArea className="flex-1">
                                        <div className="p-6 space-y-6">

                                            {/* Action Description / Reasoning */}
                                            {selectedAction.detail && (
                                                <div className="space-y-2">
                                                    <h3 className="text-sm font-semibold flex items-center gap-2">
                                                        <Activity className="w-4 h-4 text-muted-foreground" />
                                                        Execution Detail
                                                    </h3>
                                                    <div className="p-3 rounded-md bg-muted/40 text-sm leading-relaxed border">
                                                        {selectedAction.detail}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Parameters */}
                                            {selectedAction.parameters && (
                                                <div className="space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <h3 className="text-sm font-semibold flex items-center gap-2">
                                                            <Terminal className="w-4 h-4 text-muted-foreground" />
                                                            Input Parameters
                                                        </h3>
                                                    </div>
                                                    <div className="relative group">
                                                        <pre className="p-3 rounded-md bg-zinc-950 text-zinc-50 text-xs whitespace-pre-wrap break-words font-mono border border-zinc-800 w-full">
                                                            {JSON.stringify(selectedAction.parameters, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Result Data */}
                                            {selectedAction.result_data && (
                                                <div className="space-y-2">
                                                    <h3 className="text-sm font-semibold flex items-center gap-2">
                                                        <CheckCircle className="w-4 h-4 text-muted-foreground" />
                                                        Result Output
                                                    </h3>
                                                    <pre className="p-3 rounded-md bg-green-950/30 text-green-400 text-xs whitespace-pre-wrap break-words font-mono border border-green-900/50 w-full">
                                                        {JSON.stringify(selectedAction.result_data, null, 2)}
                                                    </pre>
                                                </div>
                                            )}

                                            {/* Error Details */}
                                            {selectedAction.error_details && (
                                                <div className="space-y-2">
                                                    <h3 className="text-sm font-semibold flex items-center gap-2 text-destructive">
                                                        <AlertCircle className="w-4 h-4" />
                                                        Error Log
                                                    </h3>
                                                    <pre className="p-3 rounded-md bg-red-950/30 text-red-400 text-xs whitespace-pre-wrap break-words font-mono border border-red-900/50 w-full">
                                                        {JSON.stringify(selectedAction.error_details, null, 2)}
                                                    </pre>
                                                </div>
                                            )}

                                        </div>
                                    </ScrollArea>

                                    {/* Footer Actions */}
                                    <div className="p-4 border-t bg-muted/5 flex justify-between items-center shrink-0">
                                        <div className="text-[10px] text-muted-foreground">
                                            Confidence Score: <span className="font-medium text-foreground">{Math.round((selectedAction.confidence_score || 0) * 100)}%</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {/* Context-Aware Actions */}
                                            {(selectedAction.name?.toLowerCase().includes('block') || selectedAction.action?.toLowerCase().includes('block')) && (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="h-7 text-xs gap-2"
                                                    onClick={() => alert("Checking current status... (Mock)")}
                                                >
                                                    <Activity className="w-3 h-3" />
                                                    Check Status
                                                </Button>
                                            )}

                                            {selectedAction.rollback_id && onRollback && (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="h-7 text-xs gap-2 border-orange-500/30 text-orange-500 hover:text-orange-600 hover:bg-orange-500/10"
                                                    onClick={() => handleRollbackClick(selectedAction)}
                                                >
                                                    <RotateCcw className="w-3 h-3" />
                                                    {(selectedAction.name?.toLowerCase().includes('block') || selectedAction.action?.toLowerCase().includes('block'))
                                                        ? "Unblock IP"
                                                        : "Rollback Action"}
                                                </Button>
                                            )}
                                        </div>
                                    </div>

                                </div>
                            ) : (
                                <div className="flex-1 flex items-center justify-center flex-col text-muted-foreground gap-3">
                                    <Shield className="w-12 h-12 opacity-20" />
                                    <p className="text-sm">Select an action to view full execution details</p>
                                </div>
                            )}
                        </div>
                    </div>
                </SheetContent>
            </Sheet>

            {/* Rollback Confirmation Dialog */}
            <Dialog open={rollbackConfirmOpen} onOpenChange={setRollbackConfirmOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <RotateCcw className="w-5 h-5 text-orange-500" />
                            Confirm Rollback
                        </DialogTitle>
                        <DialogDescription>
                            Are you sure you want to rollback <strong>{actionToRollback?.name || "this action"}</strong>?
                        </DialogDescription>
                    </DialogHeader>

                    <div className="p-4 rounded-md bg-muted/50 border text-sm space-y-3">
                        <div className="flex gap-2">
                            <AlertCircle className="w-4 h-4 text-orange-500 shrink-0 mt-0.5" />
                            <div className="space-y-1">
                                {(actionToRollback?.name?.toLowerCase().includes('block') || actionToRollback?.action?.toLowerCase().includes('block')) ? (
                                    <>
                                        <p className="font-medium text-foreground">Warning: Security Risk</p>
                                        <p>Unblocking this entity will immediately allow traffic from <span className="font-mono">{entityId}</span>. Ensure the threat has been mitigated or this is a false positive.</p>
                                    </>
                                ) : (
                                    <p>This will execute a compensating action to reverse the effects. Please verify the target state before proceeding.</p>
                                )}
                            </div>
                        </div>
                        {actionToRollback?.rollback_id && (
                            <div className="text-xs font-mono bg-background p-2 rounded border flex justify-between items-center">
                                <span>Rollback ID: {actionToRollback.rollback_id}</span>
                                <Badge variant="outline" className="text-[10px]">Automated</Badge>
                            </div>
                        )}
                    </div>

                    <DialogFooter>
                        <Button variant="outline" onClick={() => setRollbackConfirmOpen(false)}>Cancel</Button>
                        <Button variant="default" className="bg-orange-600 hover:bg-orange-700 text-white" onClick={confirmRollback}>
                            Confirm Rollback
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}
