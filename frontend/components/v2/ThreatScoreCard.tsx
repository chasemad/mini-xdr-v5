import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThreatScoreCardProps {
  score: number;
  factors: { label: string; score: number; type: 'positive' | 'negative' }[];
}

export default function ThreatScoreCard({ score, factors }: ThreatScoreCardProps) {
  // Calculate color based on score
  const getColor = (score: number) => {
    if (score >= 80) return "text-red-500 stroke-red-500";
    if (score >= 50) return "text-orange-500 stroke-orange-500";
    return "text-green-500 stroke-green-500";
  };

  const colorClass = getColor(score);
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <Card className="overflow-visible">
      <CardHeader className="pb-1.5 pt-3 px-4 flex flex-row items-center justify-between space-y-0">
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
      <CardContent className="flex flex-col items-center justify-center py-4 relative">
        <div className="relative w-28 h-28 flex items-center justify-center">
          {/* Background Circle */}
          <svg className="w-full h-full transform -rotate-90 overflow-visible" viewBox="0 0 112 112">
            <circle
              cx="56"
              cy="56"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-muted/20"
            />
            {/* Progress Circle */}
            <circle
              cx="56"
              cy="56"
              r={radius}
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              className={cn("transition-all duration-1000 ease-out", colorClass)}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={cn("text-4xl font-bold", colorClass.split(' ')[0])}>{score}</span>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-2 font-medium">
          {score >= 80 ? "Critical Risk" : score >= 50 ? "High Risk" : "Low Risk"}
        </p>
      </CardContent>
    </Card>
  );
}
