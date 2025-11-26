"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2, CheckCircle, XCircle, Clock,
  ChevronDown, ChevronUp, RotateCcw,
  Terminal, History, X, ExternalLink, Zap
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ActionExecutionResult } from "@/lib/actionExecutor";

interface ExecutionItem {
  actionId: string;
  actionName: string;
  status: "executing" | "completed" | "failed";
  startedAt: number;
  result?: ActionExecutionResult;
}

interface ActionExecutionPanelProps {
  executions: ExecutionItem[];
  onViewHistory?: () => void;
  onDismiss?: (actionId: string) => void;
  onRollback?: (rollbackId: string) => void;
  className?: string;
  maxVisible?: number;
}

export default function ActionExecutionPanel({
  executions,
  onViewHistory,
  onDismiss,
  onRollback,
  className,
  maxVisible = 3
}: ActionExecutionPanelProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);

  if (executions.length === 0) return null;

  const toggleExpanded = (actionId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(actionId)) {
        newSet.delete(actionId);
      } else {
        newSet.add(actionId);
      }
      return newSet;
    });
  };

  const visibleExecutions = showAll
    ? executions
    : executions.slice(0, maxVisible);

  const hasMore = executions.length > maxVisible;

  const getStatusIcon = (status: ExecutionItem["status"]) => {
    switch (status) {
      case "executing":
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getStatusBadge = (status: ExecutionItem["status"]) => {
    switch (status) {
      case "executing":
        return (
          <Badge className="text-[10px] bg-blue-500/10 text-blue-500 border-blue-500/30">
            Executing
          </Badge>
        );
      case "completed":
        return (
          <Badge className="text-[10px] bg-green-500/10 text-green-500 border-green-500/30">
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge className="text-[10px] bg-red-500/10 text-red-500 border-red-500/30">
            Failed
          </Badge>
        );
    }
  };

  const formatElapsedTime = (startedAt: number, result?: ActionExecutionResult) => {
    if (result?.duration) {
      return `${(result.duration / 1000).toFixed(1)}s`;
    }
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    return `${elapsed}s`;
  };

  return (
    <div className={cn(
      "border rounded-lg bg-card overflow-hidden",
      className
    )}>
      {/* Header */}
      <div className="px-3 py-2 border-b bg-muted/30 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">Recent Actions</span>
          <Badge variant="secondary" className="text-[10px] h-5">
            {executions.length}
          </Badge>
        </div>
        {onViewHistory && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-[10px] gap-1"
            onClick={onViewHistory}
          >
            <History className="w-3 h-3" />
            View All
          </Button>
        )}
      </div>

      {/* Execution List */}
      <ScrollArea className={cn(
        "transition-all",
        showAll && executions.length > maxVisible ? "max-h-80" : "max-h-48"
      )}>
        <div className="divide-y">
          {visibleExecutions.map((execution) => {
            const isExpanded = expandedItems.has(execution.actionId);

            return (
              <div key={execution.actionId} className="bg-background">
                {/* Main Row */}
                <div
                  className={cn(
                    "px-3 py-2 flex items-center gap-3 cursor-pointer hover:bg-muted/30 transition-colors",
                    execution.status === "executing" && "bg-blue-500/5"
                  )}
                  onClick={() => toggleExpanded(execution.actionId)}
                >
                  {getStatusIcon(execution.status)}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium truncate">
                        {execution.actionName}
                      </span>
                      {getStatusBadge(execution.status)}
                    </div>

                    {/* Execution message or progress */}
                    <div className="flex items-center gap-2 mt-0.5">
                      {execution.status === "executing" ? (
                        <span className="text-xs text-muted-foreground flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatElapsedTime(execution.startedAt)}
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground truncate">
                          {execution.result?.message || "No message"}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    {execution.status !== "executing" && onDismiss && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 opacity-50 hover:opacity-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDismiss(execution.actionId);
                        }}
                      >
                        <X className="w-3 h-3" />
                      </Button>
                    )}
                    <ChevronDown
                      className={cn(
                        "w-4 h-4 text-muted-foreground transition-transform",
                        isExpanded && "rotate-180"
                      )}
                    />
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-3 py-3 bg-muted/20 border-t space-y-3">
                    {/* Result Data */}
                    {execution.result?.data && Object.keys(execution.result.data).length > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase mb-1.5">
                          <Terminal className="w-3 h-3" />
                          Result Data
                        </div>
                        <pre className="text-[11px] font-mono bg-background p-2 rounded border overflow-auto whitespace-pre-wrap break-words text-muted-foreground w-full">
                          {JSON.stringify(execution.result.data, null, 2)}
                        </pre>
                      </div>
                    )}

                    {/* Error Details */}
                    {execution.result?.error && (
                      <div>
                        <div className="flex items-center gap-1.5 text-[10px] font-medium text-red-500 uppercase mb-1.5">
                          <XCircle className="w-3 h-3" />
                          Error
                        </div>
                        <pre className="text-[11px] font-mono bg-red-500/5 border-red-500/20 border p-2 rounded overflow-auto whitespace-pre-wrap break-words text-red-600 dark:text-red-400 w-full">
                          {execution.result.error}
                        </pre>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex items-center gap-2 pt-1">
                      {execution.result?.rollbackId && onRollback && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 text-[10px] gap-1 border-orange-500/30 text-orange-600 hover:bg-orange-500/10"
                          onClick={() => onRollback(execution.result!.rollbackId!)}
                        >
                          <RotateCcw className="w-3 h-3" />
                          Rollback
                        </Button>
                      )}

                      <div className="flex-1" />

                      <span className="text-[10px] text-muted-foreground font-mono">
                        <span suppressHydrationWarning>{new Date(execution.result?.timestamp || execution.startedAt).toLocaleTimeString()}</span>
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>

      {/* Show More/Less */}
      {hasMore && (
        <div className="px-3 py-2 border-t bg-muted/30">
          <Button
            variant="ghost"
            size="sm"
            className="w-full h-7 text-[10px] gap-1"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? (
              <>
                <ChevronUp className="w-3 h-3" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                Show {executions.length - maxVisible} More
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}

// Compact version for inline use
export function CompactExecutionIndicator({
  execution,
  onDismiss,
}: {
  execution: ExecutionItem;
  onDismiss?: () => void;
}) {
  const getStatusColor = () => {
    switch (execution.status) {
      case "executing":
        return "border-blue-500/50 bg-blue-500/5";
      case "completed":
        return "border-green-500/50 bg-green-500/5";
      case "failed":
        return "border-red-500/50 bg-red-500/5";
    }
  };

  return (
    <div className={cn(
      "flex items-center gap-2 px-3 py-1.5 rounded-md border text-sm",
      getStatusColor()
    )}>
      {execution.status === "executing" && (
        <Loader2 className="w-3 h-3 text-blue-500 animate-spin" />
      )}
      {execution.status === "completed" && (
        <CheckCircle className="w-3 h-3 text-green-500" />
      )}
      {execution.status === "failed" && (
        <XCircle className="w-3 h-3 text-red-500" />
      )}

      <span className="text-xs font-medium truncate">
        {execution.actionName}
      </span>

      {execution.status === "executing" && (
        <span className="text-[10px] text-muted-foreground font-mono ml-auto">
          {Math.round((Date.now() - execution.startedAt) / 1000)}s
        </span>
      )}

      {onDismiss && execution.status !== "executing" && (
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 ml-auto"
          onClick={onDismiss}
        >
          <X className="w-3 h-3" />
        </Button>
      )}
    </div>
  );
}
