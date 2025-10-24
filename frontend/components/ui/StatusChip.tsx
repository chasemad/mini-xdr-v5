"use client";

import React from "react";
import { CheckCircle, XCircle, Clock, AlertCircle } from "lucide-react";

type Status = "active" | "inactive" | "pending" | "error" | "success";

interface StatusChipProps {
  status: Status;
  label?: string;
  className?: string;
}

const statusConfig = {
  active: {
    label: "Active",
    icon: CheckCircle,
    bgColor: "bg-green-500/20",
    textColor: "text-green-400",
    borderColor: "border-green-500/30",
  },
  inactive: {
    label: "Inactive",
    icon: XCircle,
    bgColor: "bg-gray-500/20",
    textColor: "text-gray-400",
    borderColor: "border-gray-500/30",
  },
  pending: {
    label: "Pending",
    icon: Clock,
    bgColor: "bg-yellow-500/20",
    textColor: "text-yellow-400",
    borderColor: "border-yellow-500/30",
  },
  error: {
    label: "Error",
    icon: AlertCircle,
    bgColor: "bg-red-500/20",
    textColor: "text-red-400",
    borderColor: "border-red-500/30",
  },
  success: {
    label: "Success",
    icon: CheckCircle,
    bgColor: "bg-green-500/20",
    textColor: "text-green-400",
    borderColor: "border-green-500/30",
  },
};

export const StatusChip: React.FC<StatusChipProps> = ({
  status,
  label,
  className = "",
}) => {
  const config = statusConfig[status] || statusConfig.pending;
  const Icon = config.icon;
  const displayLabel = label || config.label;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border ${config.bgColor} ${config.textColor} ${config.borderColor} text-xs font-medium ${className}`}
    >
      <Icon className="w-3.5 h-3.5" />
      {displayLabel}
    </span>
  );
};



