"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Label } from "@/components/ui/label";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertTriangle, Shield, Activity, BrainCircuit, CheckCircle2, XCircle, AlertCircle, Clock, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface GateResult {
  gate: string;
  verdict: string;
  reason: string;
  confidence_modifier?: number;
  processing_time_ms?: number;
  details?: Record<string, any>;
}

interface OverviewTabProps {
  incident: any;
  gateResults?: GateResult[];
  escalationReasons?: string[];
  detectionMethod?: string;
}

export function OverviewTab({ incident, gateResults, escalationReasons, detectionMethod }: OverviewTabProps) {
  // Prepare data for chart
  const classProbabilities = incident.triage_note?.indicators?.enhanced_model_prediction?.class_probabilities || {};

  // Convert object to array format for the chart
  const chartData = Object.entries(classProbabilities).map(([key, prob]) => ({
    name: key.split('_').map((word: string) => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
    probability: (prob as number) * 100
  })).filter((d: any) => d.probability > 1).sort((a, b) => b.probability - a.probability); // Only show > 1%, sort by highest

  const advancedFeatures = incident.triage_note?.indicators?.phase2_advanced_features;
  const mlConfidence = incident.ml_confidence ? Math.round(incident.ml_confidence * 100) : 0;
  const riskScore = incident.risk_score ? Math.round(incident.risk_score * 100) : 0;

  // Get gate results from props or incident triage_note
  const effectiveGateResults = gateResults || incident.triage_note?.gate_results || incident.triage_note?.indicators?.gate_results || [];
  const effectiveEscalationReasons = escalationReasons || incident.triage_note?.escalation_reasons || incident.triage_note?.indicators?.escalation_reasons || [];
  const effectiveDetectionMethod = detectionMethod || incident.triage_note?.detection_method || "standard";
  const isMultiGate = effectiveDetectionMethod === "multi_gate" || effectiveGateResults.length > 0;

  // Get gate verdict icon and styling
  const getGateVerdictDisplay = (verdict: string) => {
    switch (verdict) {
      case "pass":
      case "escalate":
        return {
          icon: CheckCircle2,
          color: "text-green-500",
          bg: "bg-green-500/10",
          border: "border-green-500/30",
          label: verdict === "escalate" ? "Escalated" : "Passed",
        };
      case "fail":
      case "block":
        return {
          icon: XCircle,
          color: "text-red-500",
          bg: "bg-red-500/10",
          border: "border-red-500/30",
          label: verdict === "block" ? "Blocked" : "Failed",
        };
      case "skip":
        return {
          icon: AlertCircle,
          color: "text-yellow-500",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/30",
          label: "Skipped",
        };
      default:
        return {
          icon: AlertCircle,
          color: "text-muted-foreground",
          bg: "bg-muted/10",
          border: "border-muted/30",
          label: verdict,
        };
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Left Column */}
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-primary" />
                Threat Assessment
              </CardTitle>
              <Badge variant={riskScore > 80 ? "destructive" : "default"}>
                Risk: {riskScore}/100
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <div className="flex justify-between mb-2">
                <Label>ML Confidence</Label>
                <span className="text-sm font-medium">{mlConfidence}%</span>
              </div>
              <Progress value={mlConfidence} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                Based on {advancedFeatures?.feature_count || 79} features
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-muted rounded-lg">
                <div className="text-xs text-muted-foreground">Threat Type</div>
                <div className="font-semibold mt-1">{incident.threat_category || "Unknown"}</div>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <div className="text-xs text-muted-foreground">Escalation</div>
                <div className="font-semibold mt-1 capitalize">{incident.escalation_level || "None"}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-blue-500" />
              Event Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold">{incident.detailed_events?.length || 0}</div>
                <div className="text-xs text-muted-foreground">Total Events</div>
              </div>
              <div>
                <div className="text-2xl font-bold">
                  {incident.triage_note?.indicators?.enhanced_model_prediction?.uncertainty_score
                    ? `${(incident.triage_note.indicators.enhanced_model_prediction.uncertainty_score * 100).toFixed(1)}%`
                    : "N/A"}
                </div>
                <div className="text-xs text-muted-foreground">Uncertainty</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Right Column */}
      <div className="space-y-6">
        <Card className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <BrainCircuit className="h-5 w-5 text-purple-500" />
                    ML Analysis Details
                </CardTitle>
            </CardHeader>
            <CardContent>
                <Accordion type="single" collapsible defaultValue="prediction">
                    <AccordionItem value="prediction">
                        <AccordionTrigger>Class Probabilities</AccordionTrigger>
                        <AccordionContent>
                            <div className="h-[200px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} layout="vertical">
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" width={100} tick={{fontSize: 12}} />
                                        <Tooltip />
                                        <Bar dataKey="probability" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                                            {chartData.map((entry: any, index: number) => (
                                                <Cell key={`cell-${index}`} fill={entry.name === incident.threat_category ? "#ef4444" : "#3b82f6"} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="features">
                        <AccordionTrigger>
                            Advanced Features (Phase 2)
                            {advancedFeatures?.features_extracted && (
                                <Badge variant="outline" className="ml-2 bg-green-500/10 text-green-500 border-green-500/20">
                                    {advancedFeatures.feature_dimensions}
                                </Badge>
                            )}
                        </AccordionTrigger>
                        <AccordionContent>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between py-1 border-b">
                                    <span className="text-muted-foreground">Feature Set</span>
                                    <span>{advancedFeatures?.feature_dimensions || "Standard (79D)"}</span>
                                </div>
                                <div className="flex justify-between py-1 border-b">
                                    <span className="text-muted-foreground">Extraction Status</span>
                                    <span className={advancedFeatures?.features_extracted ? "text-green-500" : "text-yellow-500"}>
                                        {advancedFeatures?.features_extracted ? "Success" : "Pending"}
                                    </span>
                                </div>
                                <div className="flex justify-between py-1 border-b">
                                    <span className="text-muted-foreground">Feature Store</span>
                                    <span>{incident.triage_note?.indicators?.cache_hit ? "Cache Hit" : "Cache Miss"}</span>
                                </div>
                            </div>
                        </AccordionContent>
                    </AccordionItem>

                    {/* Detection Gates Section (Multi-Gate Detection) */}
                    {isMultiGate && (
                        <AccordionItem value="gates">
                            <AccordionTrigger>
                                Detection Gates
                                <Badge variant="outline" className="ml-2 bg-primary/10 text-primary border-primary/20">
                                    Multi-Gate
                                </Badge>
                            </AccordionTrigger>
                            <AccordionContent>
                                <div className="space-y-3">
                                    {/* Gate Results */}
                                    {effectiveGateResults.length > 0 ? (
                                        <div className="space-y-2">
                                            {effectiveGateResults.map((gate: GateResult, idx: number) => {
                                                const verdictDisplay = getGateVerdictDisplay(gate.verdict);
                                                const VerdictIcon = verdictDisplay.icon;
                                                return (
                                                    <div
                                                        key={idx}
                                                        className={cn(
                                                            "rounded-lg border p-3",
                                                            verdictDisplay.border,
                                                            verdictDisplay.bg
                                                        )}
                                                    >
                                                        <div className="flex items-center justify-between mb-1">
                                                            <div className="flex items-center gap-2">
                                                                <VerdictIcon className={cn("h-4 w-4", verdictDisplay.color)} />
                                                                <span className="font-medium text-sm capitalize">
                                                                    {gate.gate.replace(/_/g, " ")}
                                                                </span>
                                                            </div>
                                                            <div className="flex items-center gap-2">
                                                                <Badge variant="outline" className={cn("text-xs", verdictDisplay.color)}>
                                                                    {verdictDisplay.label}
                                                                </Badge>
                                                                {gate.confidence_modifier !== undefined && gate.confidence_modifier !== 0 && (
                                                                    <Badge variant="outline" className="text-xs">
                                                                        {gate.confidence_modifier > 0 ? "+" : ""}{(gate.confidence_modifier * 100).toFixed(0)}%
                                                                    </Badge>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-muted-foreground pl-6">{gate.reason}</p>
                                                        {gate.processing_time_ms !== undefined && (
                                                            <div className="flex items-center gap-1 text-xs text-muted-foreground/60 pl-6 mt-1">
                                                                <Clock className="h-3 w-3" />
                                                                {gate.processing_time_ms.toFixed(0)}ms
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <p className="text-sm text-muted-foreground">No gate results available</p>
                                    )}

                                    {/* Escalation Reasons */}
                                    {effectiveEscalationReasons.length > 0 && (
                                        <div className="mt-4 border-t pt-3">
                                            <div className="flex items-center gap-2 mb-2">
                                                <Zap className="h-4 w-4 text-orange-500" />
                                                <span className="text-sm font-medium text-orange-500">Escalation Triggers</span>
                                            </div>
                                            <ul className="space-y-1 pl-6">
                                                {effectiveEscalationReasons.map((reason: string, idx: number) => (
                                                    <li key={idx} className="text-xs text-muted-foreground flex items-start gap-2">
                                                        <span className="text-orange-500 mt-0.5">•</span>
                                                        {reason}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </AccordionContent>
                        </AccordionItem>
                    )}
                </Accordion>
            </CardContent>
        </Card>
      </div>
    </div>
  );
}
