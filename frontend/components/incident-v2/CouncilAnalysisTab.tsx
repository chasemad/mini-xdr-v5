"use client";

import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ShieldAlert, ShieldCheck, Search, Sparkles, Zap, Bot, ArrowRight, RefreshCw, Loader2 } from "lucide-react";

interface CouncilAnalysisTabProps {
  incident: any;
  onRefresh?: () => void;
  autoRun?: boolean; // Auto-run analysis when tab opens
}

export function CouncilAnalysisTab({ incident, onRefresh, autoRun = true }: CouncilAnalysisTabProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasAutoRun = useRef(false);

  const verdict = incident.council_verdict;
  const isThreat = verdict === "THREAT";
  const isSafe = verdict === "FALSE_POSITIVE";
  const isPending = !verdict || verdict === "PENDING";

  const geminiAnalysis = incident.gemini_analysis;
  const grokIntel = incident.grok_intel;
  const openaiRemediation = incident.openai_remediation;

  // Check for fallback modes
  const isFallbackMode =
    (geminiAnalysis?.fallback_used) ||
    (grokIntel?.status === "grok_api_not_configured") ||
    (openaiRemediation?.template_used);

  const runCouncilAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/incidents/${incident.id}/council-analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      const data = await response.json();

      if (data.success) {
        // Trigger refresh of incident data
        if (onRefresh) {
          onRefresh();
        } else {
          // Fallback: reload the page
          window.location.reload();
        }
      } else {
        setError(data.error || "Analysis failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run analysis");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Auto-run analysis when tab opens if verdict is pending and no analysis exists
  useEffect(() => {
    if (
      autoRun &&
      isPending &&
      !geminiAnalysis &&
      !grokIntel &&
      !openaiRemediation &&
      !hasAutoRun.current &&
      !isAnalyzing
    ) {
      hasAutoRun.current = true;
      console.log("🤖 Auto-triggering Council analysis for incident", incident.id);
      runCouncilAnalysis();
    }
  }, [incident.id, autoRun, isPending, geminiAnalysis, grokIntel, openaiRemediation]);

  return (
    <div className="space-y-6">
      {/* Verdict Banner with Run Analysis Button */}
      <Alert variant={isThreat ? "destructive" : isSafe ? "default" : "secondary"} className={isSafe ? "border-green-500 text-green-500" : ""}>
        <div className="flex items-start justify-between w-full">
          <div className="flex items-start">
            {isThreat ? <ShieldAlert className="h-5 w-5" /> : isSafe ? <ShieldCheck className="h-5 w-5" /> : <Search className="h-5 w-5" />}
            <div className="ml-2">
              <AlertTitle className="text-lg font-bold flex items-center gap-2">
                Council Verdict: {verdict || "PENDING"}
                <Badge variant="outline" className="ml-2">
                  Confidence: {Math.round((incident.council_confidence || 0) * 100)}%
                </Badge>
              </AlertTitle>
              <AlertDescription className="mt-2">
                {incident.council_reasoning || "Analysis in progress..."}
              </AlertDescription>
            </div>
          </div>
          <Button
            onClick={runCouncilAnalysis}
            disabled={isAnalyzing}
            variant={isPending ? "default" : "outline"}
            size="sm"
            className={isPending ? "bg-primary hover:bg-primary/90" : ""}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4 mr-2" />
                {isPending ? "Run Analysis" : "Re-analyze"}
              </>
            )}
          </Button>
        </div>
      </Alert>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertTitle>Analysis Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Fallback Mode Warning */}
      {isFallbackMode && (
        <Alert variant="default" className="border-yellow-500 bg-yellow-500/10 text-yellow-600 dark:text-yellow-400">
          <AlertTitle className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" />
            Running in Fallback Mode
          </AlertTitle>
          <AlertDescription>
            AI API keys are missing or invalid. The system is using rule-based fallback logic instead of live LLM analysis.
            Configure <code>GOOGLE_API_KEY</code>, <code>GROK_API_KEY</code>, and <code>OPENAI_API_KEY</code> for full capabilities.
          </AlertDescription>
        </Alert>
      )}

      {/* Routing Path */}
      <Card>
        <CardHeader className="pb-4">
            <CardTitle className="text-sm font-medium text-muted-foreground">Decision Path</CardTitle>
        </CardHeader>
        <CardContent>
            <div className="flex items-center gap-2 text-sm overflow-x-auto pb-2">
                <Badge variant="secondary">ML Detection</Badge>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                <Badge variant="secondary">Feature Store</Badge>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                <Badge variant={incident.ml_confidence > 0.5 ? "destructive" : "secondary"}>
                    Confidence Check ({Math.round(incident.ml_confidence * 100)}%)
                </Badge>
                {incident.routing_path?.map((step: string, idx: number) => (
                    <React.Fragment key={idx}>
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        <Badge variant="outline" className="border-primary text-primary">{step}</Badge>
                    </React.Fragment>
                ))}
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                <Badge className="bg-primary text-primary-foreground">Final Verdict</Badge>
            </div>
            <div className="mt-4 text-xs text-muted-foreground flex gap-4">
                <span>Processing Time: {incident.processing_time_ms || "N/A"}ms</span>
                <span>API Calls: {incident.api_calls_made?.length || 0}</span>
            </div>
        </CardContent>
      </Card>

      {/* Analysis Grid */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Gemini Judge */}
        <Card className="border-blue-500/20 bg-blue-500/5">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-blue-600 dark:text-blue-400">
                    <Sparkles className="h-5 w-5" />
                    Gemini Judge
                </CardTitle>
                <CardDescription>Deep Reasoning</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
                {geminiAnalysis ? (
                    <>
                        <p className="italic">"{geminiAnalysis.reasoning || "No reasoning provided"}"</p>
                        <Separator className="bg-blue-500/20 my-2" />
                        <div className="text-xs">
                            <strong>Confidence:</strong> {geminiAnalysis.confidence}
                        </div>
                    </>
                ) : <p className="text-muted-foreground">No analysis available</p>}
            </CardContent>
        </Card>

        {/* Grok Intel */}
        <Card className="border-purple-500/20 bg-purple-500/5">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-purple-600 dark:text-purple-400">
                    <Bot className="h-5 w-5" />
                    Grok Intel
                </CardTitle>
                <CardDescription>Threat Intelligence</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
                {grokIntel ? (
                    <>
                        <div>
                            <strong>Threat Actor:</strong> {grokIntel.threat_actor || "Unknown"}
                        </div>
                        <div>
                            <strong>TTPs:</strong> {grokIntel.ttps?.join(", ") || "None identified"}
                        </div>
                    </>
                ) : <p className="text-muted-foreground">No intel available</p>}
            </CardContent>
        </Card>

        {/* OpenAI Remediation */}
        <Card className="border-green-500/20 bg-green-500/5">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-600 dark:text-green-400">
                    <Zap className="h-5 w-5" />
                    OpenAI Response
                </CardTitle>
                <CardDescription>Automated Actions</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
                {openaiRemediation ? (
                    <ul className="list-disc pl-4 space-y-1">
                        {openaiRemediation.recommended_actions?.map((action: string, i: number) => (
                            <li key={i}>{action}</li>
                        )) || <li>No actions recommended</li>}
                    </ul>
                ) : <p className="text-muted-foreground">No plan available</p>}
            </CardContent>
        </Card>
      </div>
    </div>
  );
}
