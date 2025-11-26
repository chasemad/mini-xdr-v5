"use client";

import React from "react";
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
import { Separator } from "@/components/ui/separator";
import { CheckCircle, XCircle, Clock, Shield, AlertCircle, Code, Terminal, Activity, RotateCcw, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

import { Action } from "@/app/types";

interface ActionHistoryProps {
   isOpen: boolean;
   onClose: () => void;
   actions: Partial<Action>[];
   onRollback?: (actionId: string) => void;
}

export default function ActionHistorySheet({ isOpen, onClose, actions, onRollback }: ActionHistoryProps) {
   const [selectedAction, setSelectedAction] = React.useState<any>(null);

   // Sort actions by timestamp descending
   const sortedActions = [...actions].sort((a, b) => {
      const dateA = new Date(a.timestamp || a.created_at || 0).getTime();
      const dateB = new Date(b.timestamp || b.created_at || 0).getTime();
      return dateB - dateA;
   });

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
      <Sheet open={isOpen} onOpenChange={onClose}>
         <SheetContent className="w-[900px] sm:max-w-[900px] p-0 flex flex-col bg-background border-l">
            <div className="flex flex-1 overflow-hidden">
               {/* Left Side: Action List */}
               <div className="w-[280px] min-w-[280px] border-r flex flex-col bg-muted/5">
                  <SheetHeader className="p-4 border-b shrink-0">
                     <SheetTitle className="text-base flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        Action History
                     </SheetTitle>
                     <SheetDescription className="text-xs">
                        {actions.length} actions recorded
                     </SheetDescription>
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
                                 <span suppressHydrationWarning>{new Date(action.timestamp || action.created_at).toLocaleTimeString()}</span>
                                 <Badge variant="outline" className="text-[8px] h-4 px-1 font-normal opacity-70">
                                    {action.agent || action.agent_type ? 'AGENT' : 'PLAYBOOK'}
                                 </Badge>
                              </div>
                           </button>
                        ))}
                        {actions.length === 0 && (
                           <div className="p-8 text-center text-muted-foreground text-xs">
                              No actions recorded yet.
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
                              {selectedAction.agent && (
                                 <Badge variant="outline" className="text-[10px] uppercase border-primary/30 text-primary">
                                    {selectedAction.agent} Agent
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
                           <div className="p-6 space-y-8">

                              {/* Action Description / Reasoning */}
                              {selectedAction.detail && (
                                 <div className="space-y-3">
                                    <h3 className="text-sm font-semibold flex items-center gap-2">
                                       <Activity className="w-4 h-4 text-muted-foreground" />
                                       Execution Detail
                                    </h3>
                                    <div className="p-4 rounded-md bg-muted/40 text-sm leading-relaxed border overflow-x-auto">
                                       <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                                          {typeof selectedAction.detail === 'string'
                                             ? selectedAction.detail
                                             : JSON.stringify(selectedAction.detail, null, 2)}
                                       </pre>
                                    </div>
                                 </div>
                              )}

                              {/* Parameters */}
                              {(selectedAction.parameters || selectedAction.params) && (
                                 <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                       <h3 className="text-sm font-semibold flex items-center gap-2">
                                          <Terminal className="w-4 h-4 text-muted-foreground" />
                                          Input Parameters
                                       </h3>
                                       <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-6 px-2 text-[10px] gap-1 opacity-60 hover:opacity-100"
                                          onClick={() => navigator.clipboard.writeText(JSON.stringify(selectedAction.parameters || selectedAction.params, null, 2))}
                                       >
                                          <Copy className="w-3 h-3" />
                                          Copy
                                       </Button>
                                    </div>
                                    <div className="relative group">
                                       <pre className="p-4 rounded-md bg-zinc-950 text-zinc-50 text-xs font-mono border border-zinc-800 w-full overflow-x-auto max-h-[300px] overflow-y-auto">
                                          <code className="whitespace-pre">
                                             {JSON.stringify(selectedAction.parameters || selectedAction.params, null, 2)}
                                          </code>
                                       </pre>
                                    </div>
                                 </div>
                              )}

                              {/* Result Data */}
                              {selectedAction.result_data && (
                                 <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                       <h3 className="text-sm font-semibold flex items-center gap-2">
                                          <CheckCircle className="w-4 h-4 text-green-500" />
                                          Result Output
                                       </h3>
                                       <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-6 px-2 text-[10px] gap-1 opacity-60 hover:opacity-100"
                                          onClick={() => navigator.clipboard.writeText(JSON.stringify(selectedAction.result_data, null, 2))}
                                       >
                                          <Copy className="w-3 h-3" />
                                          Copy
                                       </Button>
                                    </div>
                                    <pre className="p-4 rounded-md bg-green-950/30 text-green-400 text-xs font-mono border border-green-900/50 w-full overflow-x-auto max-h-[400px] overflow-y-auto">
                                       <code className="whitespace-pre">
                                          {JSON.stringify(selectedAction.result_data, null, 2)}
                                       </code>
                                    </pre>
                                 </div>
                              )}

                              {/* Error Details */}
                              {selectedAction.error_details && (
                                 <div className="space-y-3">
                                    <div className="flex items-center justify-between">
                                       <h3 className="text-sm font-semibold flex items-center gap-2 text-destructive">
                                          <AlertCircle className="w-4 h-4" />
                                          Error Log
                                       </h3>
                                       <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-6 px-2 text-[10px] gap-1 opacity-60 hover:opacity-100"
                                          onClick={() => navigator.clipboard.writeText(JSON.stringify(selectedAction.error_details, null, 2))}
                                       >
                                          <Copy className="w-3 h-3" />
                                          Copy
                                       </Button>
                                    </div>
                                    <pre className="p-4 rounded-md bg-red-950/30 text-red-400 text-xs font-mono border border-red-900/50 w-full overflow-x-auto max-h-[300px] overflow-y-auto">
                                       <code className="whitespace-pre">
                                          {JSON.stringify(selectedAction.error_details, null, 2)}
                                       </code>
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
                           {selectedAction.rollback_id && onRollback && (
                              <Button
                                 variant="outline"
                                 size="sm"
                                 className="h-7 text-xs gap-2 border-orange-500/30 text-orange-500 hover:text-orange-600 hover:bg-orange-500/10"
                                 onClick={() => onRollback(selectedAction.rollback_id)}
                              >
                                 <RotateCcw className="w-3 h-3" />
                                 Rollback Action
                              </Button>
                           )}
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
   );
}
