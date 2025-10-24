"use client";

import React from "react";
import { AlertCircle, Shield, AlertTriangle, XCircle } from "lucide-react";

type Severity = "low" | "medium" | "high" | "critical";

interface SeverityBadgeProps {
  severity: Severity;
  className?: string;
}

const severityConfig = {
  low: {
    label: "Low",
    icon: Shield,
    bgColor: "bg-gray-500/20",
    textColor: "text-gray-400",
    borderColor: "border-gray-500/30",
  },
  medium: {
    label: "Medium",
    icon: AlertCircle,
    bgColor: "bg-yellow-500/20",
    textColor: "text-yellow-400",
    borderColor: "border-yellow-500/30",
  },
  high: {
    label: "High",
    icon: AlertTriangle,
    bgColor: "bg-orange-500/20",
    textColor: "text-orange-400",
    borderColor: "border-orange-500/30",
  },
  critical: {
    label: "Critical",
    icon: XCircle,
    bgColor: "bg-red-500/20",
    textColor: "text-red-400",
    borderColor: "border-red-500/30",
  },
};

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({
  severity,
  className = "",
}) => {
  const config = severityConfig[severity] || severityConfig.medium;
  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border ${config.bgColor} ${config.textColor} ${config.borderColor} text-xs font-medium ${className}`}
    >
      <Icon className="w-3.5 h-3.5" />
      {config.label}
    </span>
  );
};



