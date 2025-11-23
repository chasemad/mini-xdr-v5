"use client";

import React from "react";
import { CheckCircle, XCircle, Clock, Zap, AlertTriangle } from "lucide-react";

interface ExecutionResult {
  action_type: string;
  success: boolean;
  error?: string;
  result?: {
    detail?: string;
    [key: string]: any;
  };
}

interface ExecutionResultDisplayProps {
  message: string;
  executionDetails?: {
    success: boolean;
    results: ExecutionResult[];
    execution_time_ms?: number;
    steps_completed?: number;
    total_steps?: number;
    success_count?: number;
    failed_count?: number;
  };
}

export function ExecutionResultDisplay({
  message,
  executionDetails
}: ExecutionResultDisplayProps) {

  if (!executionDetails) {
    // Regular message without execution details
    return (
      <div className="whitespace-pre-wrap leading-relaxed">
        {message}
      </div>
    );
  }

  const { success, results = [], execution_time_ms, success_count = 0, failed_count = 0, total_steps = 0 } = executionDetails;

  return (
    <div className="space-y-3">
      {/* Header Status */}
      <div className={`flex items-center gap-2 ${success ? 'text-green-500' : failed_count === total_steps ? 'text-red-500' : 'text-yellow-500'}`}>
        {success ? (
          <CheckCircle className="w-5 h-5" />
        ) : failed_count === total_steps ? (
          <XCircle className="w-5 h-5" />
        ) : (
          <AlertTriangle className="w-5 h-5" />
        )}
        <span className="font-semibold">
          {success ? 'All Actions Executed Successfully!' :
           failed_count === total_steps ? 'Workflow Execution Failed' :
           'Workflow Partially Completed'}
        </span>
      </div>

      {/* Main Message */}
      <div className="text-sm text-text-muted whitespace-pre-wrap leading-relaxed">
        {message}
      </div>

      {/* Action Results */}
      {results.length > 0 && (
        <div className="space-y-2 mt-3 pt-3 border-t border-border/50">
          <div className="text-xs font-medium text-text-muted uppercase tracking-wide">
            Execution Details
          </div>
          {results.map((result, index) => {
            const actionName = result.action_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const isSuccess = result.success;

            return (
              <div key={index} className="flex items-start gap-2 text-sm">
                {isSuccess ? (
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className={`font-medium ${isSuccess ? 'text-green-400' : 'text-red-400'}`}>
                    {actionName}
                  </div>
                  {result.result?.detail && (
                    <div className="text-xs text-text-muted mt-0.5 ml-1">
                      └─ {result.result.detail}
                    </div>
                  )}
                  {result.error && (
                    <div className="text-xs text-red-400/80 mt-0.5 ml-1">
                      └─ Error: {result.error}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Summary Stats */}
      <div className="flex items-center gap-4 pt-2 mt-3 border-t border-border/50 text-xs">
        <div className="flex items-center gap-1.5">
          <Zap className="w-3.5 h-3.5 text-primary" />
          <span className="text-text-muted">
            {success_count}/{total_steps} successful
          </span>
        </div>

        {execution_time_ms !== undefined && (
          <div className="flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-text-muted" />
            <span className="text-text-muted">
              {execution_time_ms < 1000 ? `${execution_time_ms}ms` : `${(execution_time_ms / 1000).toFixed(1)}s`}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExecutionResultDisplay;
