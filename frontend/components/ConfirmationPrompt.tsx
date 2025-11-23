"use client";

import React from "react";
import { AlertCircle, CheckCircle, Clock, Shield, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ConfirmationPromptProps {
  actionPlan?: {
    action_summary: string[];
    affected_resources: string[];
    risk_level: string;
    estimated_duration: string;
    total_actions: number;
  };
  message: string;
  onApprove: () => void;
  onReject: () => void;
  isLoading?: boolean;
}

export function ConfirmationPrompt({
  actionPlan,
  message,
  onApprove,
  onReject,
  isLoading = false
}: ConfirmationPromptProps) {

  // Get risk level color and icon
  const getRiskDisplay = (riskLevel: string) => {
    switch (riskLevel) {
      case "high":
        return {
          color: "text-red-500",
          bg: "bg-red-500/10",
          border: "border-red-500/20",
          icon: <AlertTriangle className="w-5 h-5" />
        };
      case "medium":
        return {
          color: "text-yellow-500",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/20",
          icon: <AlertCircle className="w-5 h-5" />
        };
      case "low":
        return {
          color: "text-green-500",
          bg: "bg-green-500/10",
          border: "border-green-500/20",
          icon: <Shield className="w-5 h-5" />
        };
      default:
        return {
          color: "text-blue-500",
          bg: "bg-blue-500/10",
          border: "border-blue-500/20",
          icon: <AlertCircle className="w-5 h-5" />
        };
    }
  };

  const riskDisplay = actionPlan?.risk_level
    ? getRiskDisplay(actionPlan.risk_level)
    : getRiskDisplay("medium");

  return (
    <div className="bg-surface-1 border border-border rounded-lg p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={`${riskDisplay.bg} ${riskDisplay.border} border rounded-lg p-2`}>
          <div className={riskDisplay.color}>
            {riskDisplay.icon}
          </div>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-text mb-1">Action Confirmation Required</h3>
          <p className="text-sm text-text-muted">
            Review the following actions before proceeding
          </p>
        </div>
      </div>

      {/* Message */}
      <div className="bg-surface-0 rounded-md p-4">
        <p className="text-sm text-text whitespace-pre-wrap">{message}</p>
      </div>

      {/* Action Plan Details */}
      {actionPlan && (
        <div className="space-y-3">
          {/* Actions List */}
          {actionPlan.action_summary && actionPlan.action_summary.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-text mb-2 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-primary" />
                Planned Actions ({actionPlan.total_actions})
              </h4>
              <ul className="space-y-1.5">
                {actionPlan.action_summary.map((action, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm">
                    <span className="text-text-muted mt-0.5">•</span>
                    <span className="text-text flex-1">{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Affected Resources */}
          {actionPlan.affected_resources && actionPlan.affected_resources.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-text mb-2 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-warning" />
                Affected Resources
              </h4>
              <div className="flex flex-wrap gap-2">
                {actionPlan.affected_resources.map((resource, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-surface-0 border border-border rounded text-xs text-text font-mono"
                  >
                    {resource}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="flex items-center gap-4 pt-2 border-t border-border">
            {/* Risk Level */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted">Risk Level:</span>
              <span className={`text-xs font-medium ${riskDisplay.color} uppercase`}>
                {actionPlan.risk_level}
              </span>
            </div>

            {/* Duration */}
            <div className="flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-text-muted" />
              <span className="text-xs text-text-muted">Duration:</span>
              <span className="text-xs text-text font-medium">
                {actionPlan.estimated_duration}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center justify-end gap-3 pt-2">
        <Button
          variant="outline"
          onClick={onReject}
          disabled={isLoading}
          className="min-w-[100px]"
        >
          Cancel
        </Button>
        <Button
          onClick={onApprove}
          disabled={isLoading}
          className="min-w-[100px] bg-primary hover:bg-primary/90"
        >
          {isLoading ? (
            <>
              <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Executing...
            </>
          ) : (
            <>
              <CheckCircle className="w-4 h-4 mr-2" />
              Approve & Execute
            </>
          )}
        </Button>
      </div>

      {/* Warning Note */}
      {actionPlan?.risk_level === "high" && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-md p-3">
          <p className="text-xs text-red-400 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>
              High-risk action: Please verify all details carefully before proceeding.
              This action may have significant impact on your systems.
            </span>
          </p>
        </div>
      )}
    </div>
  );
}

export default ConfirmationPrompt;
