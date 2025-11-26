import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Activity, Info, Shield } from "lucide-react";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";

interface ThreatScoreCardProps {
  score: number;
  factors: { label: string; score: number; type: 'positive' | 'negative' }[];
  entityIp?: string;
  entityType?: string;
  onOpenEntityActions?: () => void;
}

export default function ThreatScoreCard({ score, factors, entityIp, entityType = "External IP", onOpenEntityActions }: ThreatScoreCardProps) {
  // Calculate color based on score
  const getColor = (score: number) => {
    if (score >= 80) return "text-red-500 stroke-red-500";
    if (score >= 50) return "text-orange-500 stroke-orange-500";
    return "text-green-500 stroke-green-500";
  };

  const colorClass = getColor(score);
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <Card className="overflow-visible">
      <CardHeader className="pb-2 pt-3 px-4 flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-sm font-medium text-muted-foreground">Threat Score</CardTitle>
        <TooltipProvider>
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <Info className="w-4 h-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent side="right" className="max-w-xs p-4 z-50">
              <div className="space-y-2">
                <h4 className="font-semibold border-b pb-1 mb-2">Score Breakdown</h4>
                {factors.map((factor, idx) => (
                  <div key={idx} className="flex justify-between text-xs gap-4">
                    <span>{factor.label}</span>
                    <span className={cn(
                      "font-mono font-bold",
                      factor.type === 'negative' ? "text-red-400" : "text-green-400"
                    )}>
                      {factor.type === 'negative' ? '+' : '-'}{factor.score}
                    </span>
                  </div>
                ))}
                {factors.length === 0 && <span className="text-muted-foreground italic">No factors available</span>}
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardHeader>
      <CardContent className="p-5 pt-2">
        <div className="flex items-center gap-5">
          {/* Gauge - Left Side */}
          <div className="flex-shrink-0">
            <div className="relative w-[110px] h-[110px] flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 96 96">
                <circle
                  cx="48"
                  cy="48"
                  r={radius}
                  stroke="currentColor"
                  strokeWidth="6"
                  fill="transparent"
                  className="text-muted/20"
                />
                <circle
                  cx="48"
                  cy="48"
                  r={radius}
                  stroke="currentColor"
                  strokeWidth="6"
                  fill="transparent"
                  strokeDasharray={circumference}
                  strokeDashoffset={offset}
                  strokeLinecap="round"
                  className={cn("transition-all duration-1000 ease-out", colorClass)}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center pt-1">
                <span className={cn("text-4xl font-bold tracking-tighter", colorClass.split(' ')[0])}>{score}</span>
                <span className="text-[10px] text-muted-foreground font-bold mt-0 text-center px-1 leading-none uppercase tracking-wider">
                  {score >= 80 ? "Critical" : score >= 50 ? "High Risk" : "Low"}
                </span>
              </div>
            </div>
          </div>

          {/* Entity Info - Right Side */}
          {entityIp && (
            <div className="flex-1 min-w-0 flex flex-col justify-center">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold mb-2">Primary Entity</div>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-muted/40 rounded-md border border-border/40 shrink-0">
                  <Shield className="w-4 h-4 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-base font-bold truncate tracking-tight text-foreground">{entityIp}</div>
                  <div className="text-xs text-muted-foreground font-medium">{entityType}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
