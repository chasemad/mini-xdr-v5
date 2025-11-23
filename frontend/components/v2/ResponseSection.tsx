"use client";

import React, { useState } from "react";
import {
    Shield, Zap, ChevronRight, ChevronDown,
    CheckCircle, AlertTriangle, Loader2,
    Play, Bot, Send, Lock, Unlock, Activity
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface ResponseSectionProps {
    activeIncident: any;
    isSourceBlocked: boolean;
    blockBadgeText?: string;
    isCheckingBlock: boolean;
    blockActionLoading: "block" | "unblock" | null;
    onBlockIp: () => Promise<void>;
    onUnblockIp: () => Promise<void>;
    onOpenMoreActions: () => void;
    isCopilotOpen: boolean;
    setIsCopilotOpen: (open: boolean) => void;
}

export default function ResponseSection({
    activeIncident,
    isSourceBlocked,
    blockBadgeText,
    isCheckingBlock,
    blockActionLoading,
    onBlockIp,
    onUnblockIp,
    onOpenMoreActions,
    isCopilotOpen,
    setIsCopilotOpen
}: ResponseSectionProps) {

    return (
        <aside className="h-full flex flex-col bg-card/30 border-l border-border overflow-hidden">

            {/* Header */}
            <div className="p-4 border-b flex items-center justify-between shrink-0 bg-background/50 backdrop-blur-sm">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    Response & Automation
                </h3>
                <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-[10px] text-muted-foreground hover:text-primary"
                    onClick={onOpenMoreActions}
                >
                    All Actions
                    <ChevronRight className="w-3 h-3 ml-1" />
                </Button>
            </div>

            <ScrollArea className="flex-1">
                <div className="p-4 space-y-6">

                    {/* Quick Actions Group */}
                    <div className="space-y-3">
                        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Recommended Actions</h4>

                        {/* Block IP Action Row */}
                        <div className="group bg-card border rounded-lg p-3 hover:border-primary/50 transition-all shadow-sm">
                            <div className="flex items-start justify-between gap-3">
                                <div className="flex items-start gap-3">
                                    <div className={cn(
                                        "p-2 rounded-md shrink-0 transition-colors",
                                        isSourceBlocked ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"
                                    )}>
                                        {isSourceBlocked ? <Lock className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h5 className="text-sm font-medium text-foreground">
                                                {isSourceBlocked ? "Unblock Source IP" : "Block Source IP"}
                                            </h5>
                                            <Badge variant="outline" className="text-[10px] h-4 px-1 bg-primary/5 text-primary border-primary/20">
                                                98% Conf.
                                            </Badge>
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                                            {isSourceBlocked
                                                ? `Source ${activeIncident.src_ip} is currently blocked.`
                                                : `Stop all traffic from ${activeIncident.src_ip}.`}
                                        </p>
                                        {blockBadgeText && (
                                            <div className="mt-2 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                                                <Activity className="w-3 h-3" />
                                                {blockBadgeText}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <Button
                                    size="sm"
                                    variant={isSourceBlocked ? "secondary" : "default"}
                                    className={cn(
                                        "h-8 text-xs shrink-0 transition-all",
                                        !isSourceBlocked && "bg-red-600 hover:bg-red-700 text-white shadow-red-500/20 shadow-lg"
                                    )}
                                    disabled={isCheckingBlock || blockActionLoading !== null}
                                    onClick={isSourceBlocked ? onUnblockIp : onBlockIp}
                                >
                                    {blockActionLoading ? (
                                        <Loader2 className="w-3 h-3 animate-spin mr-1" />
                                    ) : (
                                        isSourceBlocked ? <Unlock className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />
                                    )}
                                    {isSourceBlocked ? "Unblock" : "Execute"}
                                </Button>
                            </div>
                        </div>

                        {/* Isolate Host Action Row (Mock) */}
                        <div className="group bg-card border rounded-lg p-3 hover:border-primary/50 transition-all shadow-sm opacity-80 hover:opacity-100">
                            <div className="flex items-start justify-between gap-3">
                                <div className="flex items-start gap-3">
                                    <div className="p-2 rounded-md shrink-0 bg-orange-500/10 text-orange-500">
                                        <AlertTriangle className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <h5 className="text-sm font-medium text-foreground">Isolate Host</h5>
                                            <Badge variant="outline" className="text-[10px] h-4 px-1">
                                                85% Conf.
                                            </Badge>
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                                            Quarantine the affected host from the network.
                                        </p>
                                    </div>
                                </div>
                                <Button size="sm" variant="outline" className="h-8 text-xs shrink-0" onClick={() => alert("Action: Isolate Host")}>
                                    <Play className="w-3 h-3 mr-1" />
                                    Execute
                                </Button>
                            </div>
                        </div>
                    </div>

                    <Separator />

                    {/* Auto-Remediation Timeline */}
                    <div className="space-y-4">
                        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Auto-Remediation</h4>

                        <div className="relative pl-2">
                            {/* Vertical Line */}
                            <div className="absolute left-[15px] top-2 bottom-4 w-0.5 bg-border/50"></div>

                            {/* Step 1: Detection */}
                            <div className="relative flex gap-4 pb-6 group">
                                <div className="z-10 mt-0.5 w-7 h-7 rounded-full bg-background border-2 border-green-500 flex items-center justify-center shrink-0 shadow-sm group-hover:scale-110 transition-transform">
                                    <CheckCircle className="w-3.5 h-3.5 text-green-500" />
                                </div>
                                <div className="flex-1 pt-1">
                                    <div className="flex justify-between items-start">
                                        <h5 className="text-xs font-semibold text-foreground">Detection</h5>
                                        <span className="text-[10px] text-muted-foreground font-mono">00:00:00</span>
                                    </div>
                                    <p className="text-[11px] text-muted-foreground mt-0.5">Triggered by SSH anomaly</p>
                                </div>
                            </div>

                            {/* Step 2: Containment */}
                            <div className="relative flex gap-4 pb-6 group">
                                <div className="z-10 mt-0.5 w-7 h-7 rounded-full bg-background border-2 border-blue-500 flex items-center justify-center shrink-0 shadow-sm shadow-blue-500/20 group-hover:scale-110 transition-transform">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                                </div>
                                <div className="flex-1 pt-1">
                                    <div className="flex justify-between items-start">
                                        <h5 className="text-xs font-semibold text-blue-500">Containment</h5>
                                        <Badge variant="secondary" className="text-[9px] h-4 px-1 bg-blue-500/10 text-blue-600 hover:bg-blue-500/20">In Progress</Badge>
                                    </div>
                                    <p className="text-[11px] text-muted-foreground mt-0.5">Isolating affected subnet...</p>
                                </div>
                            </div>

                            {/* Step 3: Recovery (Pending) */}
                            <div className="relative flex gap-4 group opacity-50">
                                <div className="z-10 mt-0.5 w-7 h-7 rounded-full bg-muted border-2 border-border flex items-center justify-center shrink-0">
                                    <div className="w-2 h-2 bg-muted-foreground/30 rounded-full" />
                                </div>
                                <div className="flex-1 pt-1">
                                    <h5 className="text-xs font-semibold text-muted-foreground">Recovery</h5>
                                    <p className="text-[11px] text-muted-foreground mt-0.5">Pending containment</p>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </ScrollArea>

            {/* Copilot Section (Collapsible) */}
            <div className={cn(
                "border-t bg-background transition-all duration-300 ease-in-out flex flex-col",
                isCopilotOpen ? "h-[40%]" : "h-12"
            )}>
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => setIsCopilotOpen(!isCopilotOpen)}
                >
                    <div className="flex items-center gap-2">
                        <Bot className="w-4 h-4 text-primary" />
                        <span className="text-xs font-medium">Security Copilot</span>
                        {!isCopilotOpen && (
                            <Badge variant="secondary" className="text-[9px] h-4 px-1">1 New</Badge>
                        )}
                    </div>
                    <ChevronDown className={cn("w-4 h-4 text-muted-foreground transition-transform", !isCopilotOpen && "rotate-180")} />
                </div>

                {isCopilotOpen && (
                    <>
                        <ScrollArea className="flex-1 p-4 pt-0">
                            <div className="space-y-4">
                                <div className="flex gap-3">
                                    <Avatar className="w-6 h-6 shrink-0 mt-1">
                                        <AvatarFallback className="bg-primary text-primary-foreground text-[10px]">AI</AvatarFallback>
                                    </Avatar>
                                    <div className="bg-muted/50 p-3 rounded-lg rounded-tl-none text-xs leading-relaxed">
                                        <p>Based on the logs, this appears to be a brute force attack. I recommend blocking the IP immediately to prevent further access attempts.</p>
                                    </div>
                                </div>
                            </div>
                        </ScrollArea>
                        <div className="p-3 border-t bg-background/50">
                            <div className="flex gap-2">
                                <Input className="h-8 text-xs bg-background" placeholder="Ask Copilot..." />
                                <Button size="icon" className="h-8 w-8 shrink-0">
                                    <Send className="w-3 h-3" />
                                </Button>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </aside>
    );
}
