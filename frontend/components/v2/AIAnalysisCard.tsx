import React from 'react';
import { BrainCircuit, ChevronRight, Sparkles, AlertCircle, ShieldCheck, ArrowRight, RefreshCw, Loader2, Zap, Shield, CheckCircle2, XCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

interface GateResult {
  gate: string;
  verdict: string;
  reason: string;
  confidence_modifier?: number;
  processing_time_ms?: number;
  details?: Record<string, any>;
}

interface AIAnalysisCardProps {
  triageNote: {
    summary?: string;
    recommendation?: string;
    rationale?: string[] | string;
    confidence?: number;
    anomaly_score?: number;
    threat_class?: number;
    event_count?: number;
    key_events_count?: number;
    last_updated?: string;
    detection_method?: string;
    gate_results?: GateResult[];
    escalation_reasons?: string[];
  } | null;
  analysis?: {
    summary?: string;
    recommendation?: string;
    rationale?: string[];
    attack_phases?: string[];
    techniques_observed?: string[];
    key_events?: any[];
    event_statistics?: {
      total_events?: number;
      key_events_count?: number;
      login_success?: number;
      login_failed?: number;
      file_transfers?: number;
    };
    confidence?: number;
    escalation_level?: string;
    generated_at?: string;
  } | null;
  // New detection metadata props
  gateResults?: GateResult[];
  escalationReasons?: string[];
  detectionMethod?: string;
  isLoading?: boolean;
  onRefresh?: () => void;
  onShowDeepAnalysis: () => void;
  className?: string;
}

export default function AIAnalysisCard({
  triageNote,
  analysis,
  gateResults,
  escalationReasons,
  detectionMethod,
  isLoading,
  onRefresh,
  onShowDeepAnalysis,
  className
}: AIAnalysisCardProps) {
  // Use enhanced analysis if available, otherwise fall back to triage note
  const summary = analysis?.summary || triageNote?.summary || "Automated analysis detected anomalous behavior consistent with known attack patterns.";
  const recommendation = analysis?.recommendation || triageNote?.recommendation || "Investigate source IP and block if malicious.";

  // Get detection metadata from props or triage_note
  const effectiveDetectionMethod = detectionMethod || triageNote?.detection_method || "standard";
  const effectiveGateResults = gateResults || triageNote?.gate_results || [];
  const effectiveEscalationReasons = escalationReasons || triageNote?.escalation_reasons || [];

  // Format detection method for display
  const detectionMethodLabels: Record<string, { label: string; variant: "default" | "secondary" | "outline" | "destructive" }> = {
    multi_gate: { label: "Multi-Gate", variant: "default" },
    standard: { label: "Standard ML", variant: "secondary" },
    local_ml_openai_enhanced: { label: "ML + OpenAI", variant: "default" },
  };
  const methodDisplay = detectionMethodLabels[effectiveDetectionMethod] || { label: effectiveDetectionMethod, variant: "secondary" as const };

  // Build rationale from available data
  const rationale: string[] = [];

  // First try to use enhanced analysis rationale
  if (analysis?.rationale && Array.isArray(analysis.rationale)) {
    rationale.push(...analysis.rationale);
  } else if (triageNote?.rationale) {
    if (Array.isArray(triageNote.rationale)) {
      rationale.push(...triageNote.rationale);
    } else if (typeof triageNote.rationale === 'string') {
      rationale.push(triageNote.rationale);
    }
  }

  // Add attack phase info if available
  if (analysis?.attack_phases && analysis.attack_phases.length > 0) {
    const phaseNames: Record<string, string> = {
      reconnaissance: "Reconnaissance",
      credential_attack: "Credential Attack",
      initial_access: "Initial Access",
      execution: "Command Execution",
      persistence: "Persistence",
      collection: "Data Collection",
      exfiltration: "Exfiltration"
    };
    const phases = analysis.attack_phases.map(p => phaseNames[p] || p).join(" → ");
    rationale.push(`Attack chain: ${phases}`);
  }

  // Add techniques if available
  if (analysis?.techniques_observed && analysis.techniques_observed.length > 0) {
    rationale.push(`Techniques: ${analysis.techniques_observed.slice(0, 3).join(", ")}`);
  }

  // Add event statistics if available
  if (analysis?.event_statistics) {
    const stats = analysis.event_statistics;
    if (stats.login_success && stats.login_success > 0) {
      rationale.push(`⚠️ ${stats.login_success} successful authentication(s) detected`);
    }
    if (stats.file_transfers && stats.file_transfers > 0) {
      rationale.push(`📁 ${stats.file_transfers} file transfer(s) detected`);
    }
  }

  // Fallback to building from triage note fields
  if (rationale.length === 0) {
    const confidence = analysis?.confidence || triageNote?.confidence;
    if (confidence) {
      const level = confidence >= 0.7 ? "high" : confidence >= 0.4 ? "moderate" : "low";
      rationale.push(`The attack has a ${level} machine learning confidence of ${(confidence * 100).toFixed(1)}%.`);
    }
    if (triageNote?.threat_class !== undefined) {
      const threatTypes = ['Normal', 'DDoS/DoS', 'Network Recon', 'Brute Force', 'Web Attack', 'Malware', 'APT'];
      rationale.push(`Threat classification: ${threatTypes[triageNote.threat_class] || 'Unknown'}`);
    }

    // Final fallback
    if (rationale.length === 0) {
      rationale.push("Multiple connection attempts to common ports indicate a targeted brute-force attempt.");
    }
  }

  // Event count display
  const eventCount = analysis?.event_statistics?.total_events || triageNote?.event_count || 0;
  const keyEventCount = analysis?.event_statistics?.key_events_count || triageNote?.key_events_count || 0;

  return (
    <Card className={cn(
      "flex flex-col overflow-hidden border-primary/20 shadow-lg relative group",
      "bg-gradient-to-b from-card to-card/95 dark:from-[#1a1f2e] dark:to-[#0f1219]",
      className
    )}>
      {/* Ambient Glow Effect */}
      <div className="absolute top-0 right-0 w-[300px] h-[300px] bg-primary/5 blur-[100px] rounded-full pointer-events-none -z-10" />

      <CardHeader className="pb-2.5 pt-3 px-4 border-b border-border/50 bg-muted/20">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-bold flex items-center gap-2 text-primary">
            <div className="p-1.5 bg-primary/10 rounded-md ring-1 ring-primary/20">
              <BrainCircuit className="w-4 h-4" />
            </div>
            AI Threat Analysis
          </CardTitle>
          <div className="flex items-center gap-2">
            {/* Detection Method Badge */}
            <Badge variant={methodDisplay.variant} className="text-[10px] font-medium">
              <Shield className="w-2.5 h-2.5 mr-1" />
              {methodDisplay.label}
            </Badge>
            {eventCount > 0 && (
              <Badge variant="outline" className="text-[10px] font-medium">
                {keyEventCount > 0 ? `${keyEventCount} key / ` : ""}{eventCount} events
              </Badge>
            )}
            {onRefresh && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                disabled={isLoading}
                className="h-7 w-7 p-0"
                title="Refresh analysis with latest events"
              >
                {isLoading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5" />
                )}
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 min-h-0 p-4 space-y-3 overflow-y-auto scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent">
        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            <span className="text-sm">Analyzing events...</span>
          </div>
        ) : (
          <>
            {/* Summary Section */}
            <div className="space-y-1.5">
              <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Situation Summary</h4>
              <p className="text-sm text-foreground leading-snug font-medium">
                {summary}
              </p>
            </div>

            {/* Recommendation Box */}
            <div className="rounded-md border border-primary/20 bg-primary/5 p-2.5 relative overflow-hidden">
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary" />
              <h4 className="text-[10px] font-bold text-primary mb-1 flex items-center gap-1.5 uppercase tracking-wider">
                <ShieldCheck className="w-3 h-3 shrink-0" />
                Recommended Action
              </h4>
              <p className="text-sm font-semibold text-foreground/90">
                {recommendation}
              </p>
            </div>

            {/* Rationale Section */}
            <div className="space-y-1.5">
              <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <AlertCircle className="w-3 h-3 shrink-0" />
                Key Risk Indicators
              </h4>
              <ul className="space-y-1.5">
                {(Array.isArray(rationale) ? rationale : []).map((item, idx) => (
                  <li key={idx} className="text-xs text-muted-foreground flex gap-2 items-start group-hover:text-foreground/80 transition-colors">
                    <span className="text-primary mt-1 shrink-0 w-1 h-1 rounded-full bg-current" />
                    <span className="leading-snug">{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Escalation Reasons (from multi-gate detection) */}
            {effectiveEscalationReasons.length > 0 && (
              <div className="space-y-1.5">
                <h4 className="text-[10px] font-bold text-destructive uppercase tracking-wider flex items-center gap-1.5">
                  <Zap className="w-3 h-3 shrink-0" />
                  Escalation Triggers
                </h4>
                <div className="rounded-md border border-destructive/20 bg-destructive/5 p-2">
                  <ul className="space-y-1">
                    {effectiveEscalationReasons.map((reason, idx) => (
                      <li key={idx} className="text-xs text-destructive/80 flex gap-2 items-start">
                        <span className="text-destructive mt-1 shrink-0">•</span>
                        <span className="leading-snug">{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* Gate Summary (if multi-gate detection) */}
            {effectiveGateResults.length > 0 && (
              <div className="space-y-1.5">
                <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <Shield className="w-3 h-3 shrink-0" />
                  Detection Gates
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {effectiveGateResults.map((gate, idx) => {
                    const isPassed = gate.verdict === "pass" || gate.verdict === "escalate";
                    const isFailed = gate.verdict === "fail" || gate.verdict === "block";
                    return (
                      <Badge
                        key={idx}
                        variant="outline"
                        className={cn(
                          "text-[9px] py-0.5 px-1.5",
                          isPassed && "border-green-500/50 text-green-600 bg-green-500/5",
                          isFailed && "border-red-500/50 text-red-600 bg-red-500/5",
                          !isPassed && !isFailed && "border-yellow-500/50 text-yellow-600 bg-yellow-500/5"
                        )}
                      >
                        {isPassed ? (
                          <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" />
                        ) : isFailed ? (
                          <XCircle className="w-2.5 h-2.5 mr-0.5" />
                        ) : null}
                        {gate.gate.replace("_", " ")}
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>

      <CardFooter className="p-4 border-t border-border/50 bg-background/50 backdrop-blur-sm">
        <Button
          className="w-full bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm transition-all group/btn"
          onClick={onShowDeepAnalysis}
        >
          <BrainCircuit className="w-4 h-4 mr-2" />
          Launch Deep Analysis
          <ArrowRight className="w-4 h-4 ml-auto opacity-70 group-hover/btn:translate-x-1 transition-transform" />
        </Button>
      </CardFooter>
    </Card>
  );
}
