"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle, Play, ShieldAlert, Info, User, Activity, RotateCcw,
  Loader2, CheckCircle, XCircle, Clock, ExternalLink, History, Terminal
} from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { ActionExecutionResult } from "@/lib/actionExecutor";

// Execution state type
export type ExecutionStatus = "idle" | "executing" | "completed" | "failed";

interface ActionConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void | Promise<void>;
  onRetry?: () => void | Promise<void>;
  onViewHistory?: () => void;
  action: Partial<import("@/app/types").Action> | null;
  executionStatus?: ExecutionStatus;
  executionResult?: ActionExecutionResult | null;
  executionStartTime?: number;
}

export default function ActionConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  onRetry,
  onViewHistory,
  action,
  executionStatus = "idle",
  executionResult,
  executionStartTime,
}: ActionConfirmationModalProps) {
  const [isConfirmed, setIsConfirmed] = React.useState(false);
  const [elapsedTime, setElapsedTime] = React.useState(0);

  // Update elapsed time during execution
  React.useEffect(() => {
    if (executionStatus === "executing" && executionStartTime) {
      const interval = setInterval(() => {
        setElapsedTime(Math.round((Date.now() - executionStartTime) / 1000));
      }, 100);
      return () => clearInterval(interval);
    } else {
      setElapsedTime(0);
    }
  }, [executionStatus, executionStartTime]);

  // Reset confirmation when modal opens/closes
  React.useEffect(() => {
    if (!isOpen) {
      setIsConfirmed(false);
    }
  }, [isOpen]);

  if (!action) return null;

  // Derive display values from new or legacy fields
  const actionName = action.name || action.action || action.action_name || "Unknown Action";
  const description = action.description || action.detail || "No description available.";
  const isDangerous = action.riskLevel === 'critical' || action.riskLevel === 'high' || (action as any).isDangerous;
  const requiresApproval = action.requiresApproval;
  const riskLevel = action.riskLevel || (isDangerous ? 'high' : 'low');

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-100 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-900';
      case 'high': return 'bg-orange-100 text-orange-700 border-orange-200 dark:bg-orange-950/30 dark:text-orange-400 dark:border-orange-900';
      case 'medium': return 'bg-yellow-100 text-yellow-700 border-yellow-200 dark:bg-yellow-950/30 dark:text-yellow-400 dark:border-yellow-900';
      default: return 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-400 dark:border-blue-900';
    }
  };

  const handleClose = () => {
    // Don't allow closing during execution
    if (executionStatus === "executing") return;
    onClose();
  };

  const handleConfirm = async () => {
    await onConfirm();
  };

  const handleRetry = async () => {
    if (onRetry) {
      await onRetry();
    } else {
      await onConfirm();
    }
  };

  // Get dialog title based on status
  const getDialogTitle = () => {
    switch (executionStatus) {
      case "executing":
        return "Executing Action";
      case "completed":
        return "Action Completed";
      case "failed":
        return "Action Failed";
      default:
        return "Confirm Action Execution";
    }
  };

  // Get dialog icon based on status
  const getDialogIcon = () => {
    switch (executionStatus) {
      case "executing":
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "failed":
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return isDangerous ? (
          <AlertTriangle className="w-5 h-5 text-orange-500" />
        ) : (
          <ShieldAlert className="w-5 h-5 text-blue-500" />
        );
    }
  };

  // Render execution progress
  const renderExecutionProgress = () => (
    <div className="py-8 flex flex-col items-center gap-4">
      <div className="relative">
        <div className="w-20 h-20 rounded-full bg-blue-500/10 flex items-center justify-center">
          <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-background border-2 border-blue-500 flex items-center justify-center">
          <Clock className="w-4 h-4 text-blue-500" />
        </div>
      </div>
      <div className="text-center">
        <h3 className="font-semibold text-lg text-foreground">Executing {actionName}</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Please wait while the action is being processed...
        </p>
        <div className="mt-3 flex items-center justify-center gap-2 text-sm font-mono text-muted-foreground">
          <Clock className="w-4 h-4" />
          Elapsed: {elapsedTime}s
        </div>
      </div>
      <div className="w-full max-w-xs h-1.5 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-blue-500 rounded-full animate-pulse" style={{ width: "100%" }} />
      </div>
    </div>
  );

  // Render success result
  const renderSuccessResult = () => (
    <div className="py-6 flex flex-col items-center gap-4">
      <div className="w-20 h-20 rounded-full bg-green-500/10 flex items-center justify-center">
        <CheckCircle className="w-10 h-10 text-green-500" />
      </div>
      <div className="text-center">
        <h3 className="font-semibold text-lg text-green-600 dark:text-green-400">
          Action Completed Successfully
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          {executionResult?.message || `${actionName} has been executed.`}
        </p>
        {executionResult?.duration && (
          <p className="text-xs text-muted-foreground mt-1 font-mono">
            Duration: {(executionResult.duration / 1000).toFixed(2)}s
          </p>
        )}
      </div>

      {/* Result Data */}
      {executionResult?.data && Object.keys(executionResult.data).length > 0 && (
        <div className="w-full mt-2">
          <div className="flex items-center gap-2 text-xs font-medium text-foreground mb-2">
            <Terminal className="w-3 h-3" />
            Result Details
          </div>
          <ScrollArea className="h-40">
            <div className="bg-muted/50 border rounded-lg p-3">
              <pre className="text-xs text-muted-foreground font-mono whitespace-pre-wrap">
                {JSON.stringify(executionResult.data, null, 2)}
              </pre>
            </div>
          </ScrollArea>
        </div>
      )}

      {/* Rollback Info */}
      {executionResult?.rollbackId && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-blue-500/5 px-4 py-2 rounded-md border border-blue-500/20">
          <RotateCcw className="w-3 h-3 text-blue-500" />
          <span>Rollback available:</span>
          <code className="font-mono text-blue-600 dark:text-blue-400">{executionResult.rollbackId}</code>
        </div>
      )}
    </div>
  );

  // Render failure result
  const renderFailureResult = () => (
    <div className="py-6 flex flex-col items-center gap-4">
      <div className="w-20 h-20 rounded-full bg-red-500/10 flex items-center justify-center">
        <XCircle className="w-10 h-10 text-red-500" />
      </div>
      <div className="text-center">
        <h3 className="font-semibold text-lg text-red-600 dark:text-red-400">
          Action Failed
        </h3>
        <p className="text-sm text-muted-foreground mt-1">
          {executionResult?.message || `${actionName} could not be completed.`}
        </p>
      </div>

      {/* Error Details */}
      {executionResult?.error && (
        <div className="w-full mt-2">
          <div className="flex items-center gap-2 text-xs font-medium text-red-600 dark:text-red-400 mb-2">
            <AlertTriangle className="w-3 h-3" />
            Error Details
          </div>
          <ScrollArea className="h-32">
            <div className="bg-red-500/5 border border-red-500/20 rounded-lg p-3">
              <pre className="text-xs text-red-600 dark:text-red-400 font-mono whitespace-pre-wrap">
                {executionResult.error}
              </pre>
            </div>
          </ScrollArea>
        </div>
      )}
    </div>
  );

  // Render confirmation form
  const renderConfirmationForm = () => (
    <div className="space-y-4 my-2">
      {/* Main Action Card */}
      <div className="bg-muted/30 border rounded-lg p-4">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-semibold text-lg">{actionName}</h3>
          <Badge variant="outline" className={getRiskColor(riskLevel)}>
            {riskLevel.toUpperCase()} RISK
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground mb-4">{description}</p>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">Target Scope</span>
            <div className="font-medium mt-0.5 flex items-center gap-2">
              {action.scope?.type === 'entity' && <User className="w-3 h-3 text-muted-foreground" />}
              {action.scope?.type === 'segment' && <Activity className="w-3 h-3 text-muted-foreground" />}
              {action.scope?.target || "Global / Unspecified"}
            </div>
          </div>
          <div>
            <span className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">Agent</span>
            <div className="font-medium mt-0.5">{action.agent || "System"}</div>
          </div>
          <div className="col-span-2">
            <span className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">Expected Impact</span>
            <div className="font-medium mt-0.5 text-foreground/90">
              {action.expectedImpact || "Standard execution impact."}
            </div>
          </div>
        </div>
      </div>

      {/* Parameters Preview */}
      {action.parameters && Object.keys(action.parameters).length > 0 && (
        <div className="text-xs">
          <span className="font-semibold text-muted-foreground mb-1 block">Parameters</span>
          <div className="bg-muted p-2 rounded border font-mono text-muted-foreground">
            {Object.entries(action.parameters).map(([k, v]) => (
              <div key={k}><span className="text-foreground">{k}:</span> {String(v)}</div>
            ))}
          </div>
        </div>
      )}

      {/* Rollback Info */}
      {action.rollbackPath && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-blue-500/5 p-2 rounded border border-blue-500/20">
          <RotateCcw className="w-3 h-3 text-blue-500" />
          <span>Rollback available: <span className="font-mono text-blue-600 dark:text-blue-400">{action.rollbackPath}</span></span>
        </div>
      )}

      {/* Warnings */}
      {isDangerous && (
        <div className="flex items-start gap-2 p-3 bg-red-500/5 border border-red-500/20 rounded-md text-red-700 dark:text-red-400 text-sm">
          <Info className="w-4 h-4 mt-0.5 shrink-0" />
          <p>This action is disruptive. It may affect service availability or data integrity. Ensure you have authorization.</p>
        </div>
      )}

      {requiresApproval && (
        <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-md text-yellow-700 dark:text-yellow-400 text-sm">
          <Info className="w-4 h-4 mt-0.5 shrink-0" />
          <p>Secondary approval will be requested immediately after initiation.</p>
        </div>
      )}

      {isDangerous && (
        <div className="flex items-center space-x-2 py-2">
          <Checkbox
            id="confirm-hazardous"
            checked={isConfirmed}
            onCheckedChange={(c) => setIsConfirmed(!!c)}
          />
          <Label htmlFor="confirm-hazardous" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
            I understand the risks and want to proceed
          </Label>
        </div>
      )}
    </div>
  );

  // Render footer buttons
  const renderFooter = () => {
    switch (executionStatus) {
      case "executing":
        return (
          <Button
            variant="outline"
            disabled
            className="bg-background text-muted-foreground"
          >
            <Loader2 className="w-4 h-4 animate-spin mr-2" />
            Processing...
          </Button>
        );

      case "completed":
        return (
          <div className="flex gap-2">
            {onViewHistory && (
              <Button
                variant="outline"
                onClick={onViewHistory}
                className="gap-2"
              >
                <History className="w-4 h-4" />
                View History
              </Button>
            )}
            <Button
              onClick={handleClose}
              className="bg-green-600 hover:bg-green-700 text-white gap-2"
            >
              <CheckCircle className="w-4 h-4" />
              Done
            </Button>
          </div>
        );

      case "failed":
        return (
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={handleClose}
            >
              Close
            </Button>
            <Button
              onClick={handleRetry}
              className="gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Retry
            </Button>
          </div>
        );

      default:
        return (
          <>
            <DialogClose asChild>
              <Button variant="outline" onClick={handleClose}>Cancel</Button>
            </DialogClose>
            <Button
              onClick={handleConfirm}
              disabled={isDangerous && !isConfirmed}
              variant={isDangerous ? "destructive" : "default"}
              className="gap-2"
            >
              <Play className="w-4 h-4" />
              {requiresApproval ? "Request Approval" : "Execute Action"}
            </Button>
          </>
        );
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className={cn(
        "sm:max-w-[600px]",
        executionStatus === "executing" && "sm:max-w-[450px]"
      )}>
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className={cn(
              "p-2 rounded-full",
              executionStatus === "executing" && "bg-blue-100 dark:bg-blue-950/30",
              executionStatus === "completed" && "bg-green-100 dark:bg-green-950/30",
              executionStatus === "failed" && "bg-red-100 dark:bg-red-950/30",
              executionStatus === "idle" && (isDangerous ? 'bg-orange-100 dark:bg-orange-950/30' : 'bg-blue-100 dark:bg-blue-950/30')
            )}>
              {getDialogIcon()}
            </div>
            <div>
              <DialogTitle>{getDialogTitle()}</DialogTitle>
              {executionStatus === "idle" && (
                <DialogDescription className="mt-1">
                  Review the action details below before proceeding.
                </DialogDescription>
              )}
            </div>
          </div>
        </DialogHeader>

        {executionStatus === "executing" && renderExecutionProgress()}
        {executionStatus === "completed" && renderSuccessResult()}
        {executionStatus === "failed" && renderFailureResult()}
        {executionStatus === "idle" && renderConfirmationForm()}

        <DialogFooter className="gap-2 sm:gap-0">
          {renderFooter()}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
