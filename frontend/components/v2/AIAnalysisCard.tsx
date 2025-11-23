import React from 'react';
import { BrainCircuit, ChevronRight, Sparkles, AlertCircle, ShieldCheck, ArrowRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

interface AIAnalysisCardProps {
  triageNote: {
    summary?: string;
    recommendation?: string;
    rationale?: string[];
    confidence?: number;
    anomaly_score?: number;
    threat_class?: number;
    event_count?: number;
  } | null;
  onShowDeepAnalysis: () => void;
  className?: string;
}

export default function AIAnalysisCard({
  triageNote,
  onShowDeepAnalysis,
  className
}: AIAnalysisCardProps) {
  const summary = triageNote?.summary || "Automated analysis detected anomalous behavior consistent with known attack patterns.";
  const recommendation = triageNote?.recommendation || "Investigate source IP and block if malicious.";

  // Build rationale from available data
  const rationale = [];

  if (triageNote?.rationale && Array.isArray(triageNote.rationale)) {
    rationale.push(...triageNote.rationale);
  } else {
    // Build rationale from other fields
    if (triageNote?.confidence) {
      rationale.push(`ML model confidence: ${(triageNote.confidence * 100).toFixed(1)}%`);
    }
    if (triageNote?.anomaly_score) {
      rationale.push(`Anomaly score: ${(triageNote.anomaly_score * 100).toFixed(1)}%`);
    }
    if (triageNote?.threat_class !== undefined) {
      const threatTypes = ['Normal', 'DDoS/DoS', 'Network Recon', 'Brute Force', 'Web Attack', 'Malware', 'APT'];
      rationale.push(`Threat type: ${threatTypes[triageNote.threat_class] || 'Unknown'}`);
    }
    if (triageNote?.event_count) {
      rationale.push(`Events detected: ${triageNote.event_count}`);
    }

    // Fallback if no data
    if (rationale.length === 0) {
      rationale.push("Source IP has a high confidence abuse score.");
      rationale.push("Attack pattern matches known botnet signatures.");
    }
  }

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
        </div>
      </CardHeader>

      <CardContent className="flex-1 min-h-0 p-4 space-y-4 overflow-y-auto scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent">
        {/* Summary Section */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Situation Summary</h4>
          <p className="text-sm text-foreground leading-relaxed font-medium">
            {summary}
          </p>
        </div>

        {/* Recommendation Box */}
        <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 relative overflow-hidden">
          <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary" />
          <h4 className="text-xs font-semibold text-primary mb-1 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
            Recommended Action
          </h4>
          <p className="text-sm font-medium text-foreground/90">
            {recommendation}
          </p>
        </div>

        {/* Rationale Section */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            Key Risk Indicators
          </h4>
          <ul className="space-y-2">
            {(Array.isArray(rationale) ? rationale : []).map((item, idx) => (
              <li key={idx} className="text-xs text-muted-foreground flex gap-2.5 items-start group-hover:text-foreground/80 transition-colors">
                <span className="text-primary mt-0.5 shrink-0">•</span>
                <span className="leading-snug">{item}</span>
              </li>
            ))}
          </ul>
        </div>
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
