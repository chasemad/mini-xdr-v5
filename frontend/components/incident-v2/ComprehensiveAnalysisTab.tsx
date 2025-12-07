"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import {
    BrainCircuit,
    Activity,
    Shield,
    Zap,
    Search,
    Target,
    ListChecks,
    Loader2,
    CheckCircle2,
    XCircle,
    AlertTriangle,
    Bot,
    Eye,
    FileSearch
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getApiKey } from "@/app/utils/api";

interface ComprehensiveAnalysisTabProps {
    incident: any;
}

export function ComprehensiveAnalysisTab({ incident }: ComprehensiveAnalysisTabProps) {
    const [investigations, setInvestigations] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Stream investigation results for this incident
    useEffect(() => {
        const eventSource = new EventSource(
            `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/incidents/${incident.id}/investigations/stream`
        );

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === "connected") {
                    console.log("Connected to investigation stream");
                    setIsLoading(false);
                } else if (data.type === "investigation_result") {
                    setInvestigations((prev) => {
                        // Check if result already exists to avoid duplicates/updates
                        const index = prev.findIndex((i) => i.id === data.id);
                        if (index >= 0) {
                            const newInvestigations = [...prev];
                            newInvestigations[index] = data;
                            return newInvestigations;
                        }
                        return [...prev, data];
                    });
                } else if (data.type === "error") {
                    console.error("Stream error:", data.message);
                }
            } catch (e) {
                console.error("Error parsing SSE message:", e);
            }
        };

        eventSource.onerror = (err) => {
            console.error("EventSource failed:", err);
            eventSource.close();
            setIsLoading(false);
        };

        return () => {
            eventSource.close();
        };
    }, [incident.id]);

    // Extract ML data
    const mlPrediction = incident.triage_note?.indicators?.enhanced_model_prediction;
    const classProbabilities = mlPrediction?.class_probabilities || {};
    const topPredictions = Object.entries(classProbabilities)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 4)
        .map(([name, prob]) => ({
            name: name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
            probability: (prob as number) * 100
        }));

    // Extract LangChain/AI Agent data
    const langchainVerdict = incident.triage_note?.langchain_verdict;
    const langchainReasoning = incident.triage_note?.langchain_reasoning;
    const langchainActions = incident.triage_note?.langchain_actions || [];

    // Extract Council data
    const geminiAnalysis = incident.gemini_analysis;
    const grokIntel = incident.grok_intel;
    const openaiRemediation = incident.openai_remediation;

    // Confidence scores
    const mlConfidence = incident.ml_confidence ? Math.round(incident.ml_confidence * 100) : 0;
    const councilConfidence = incident.council_confidence ? Math.round(incident.council_confidence * 100) : 0;

    // Risk assessment
    const riskScore = incident.risk_score ? Math.round(incident.risk_score * 100) : 0;
    const severityColor = riskScore > 80 ? "text-red-500" : riskScore > 50 ? "text-orange-500" : "text-yellow-500";

    return (
        <div className="space-y-6">
            {/* Executive Summary Card */}
            <Card className="border-primary/30 bg-gradient-to-br from-primary/5 to-background">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <BrainCircuit className="h-5 w-5 text-primary" />
                        Executive AI Analysis Summary
                    </CardTitle>
                    <CardDescription>Comprehensive AI-driven threat assessment and recommendations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Key Metrics */}
                    <div className="grid grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-background/80 rounded-lg border">
                            <div className={cn("text-3xl font-bold", severityColor)}>{riskScore}</div>
                            <div className="text-xs text-muted-foreground mt-1">Risk Score</div>
                            <Progress value={riskScore} className="h-1.5 mt-2" />
                        </div>
                        <div className="text-center p-4 bg-background/80 rounded-lg border">
                            <div className="text-3xl font-bold text-blue-500">{mlConfidence}%</div>
                            <div className="text-xs text-muted-foreground mt-1">ML Confidence</div>
                            <Progress value={mlConfidence} className="h-1.5 mt-2" />
                        </div>
                        <div className="text-center p-4 bg-background/80 rounded-lg border">
                            <div className="text-3xl font-bold text-purple-500">{councilConfidence}%</div>
                            <div className="text-xs text-muted-foreground mt-1">Council Consensus</div>
                            <Progress value={councilConfidence} className="h-1.5 mt-2" />
                        </div>
                    </div>

                    {/* Threat Classification */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-muted/50 rounded-lg border">
                            <div className="text-sm text-muted-foreground mb-2">Threat Category</div>
                            <Badge variant="destructive" className="text-base font-semibold">
                                {incident.threat_category || "Unknown"}
                            </Badge>
                        </div>
                        <div className="p-4 bg-muted/50 rounded-lg border">
                            <div className="text-sm text-muted-foreground mb-2">Escalation Level</div>
                            <Badge variant="outline" className="text-base font-semibold capitalize">
                                {incident.escalation_level || "None"}
                            </Badge>
                        </div>
                    </div>

                    {/* Final Verdict */}
                    {langchainVerdict && (
                        <Alert className={langchainVerdict === "THREAT" ? "border-red-500" : "border-green-500"}>
                            <AlertTitle className="flex items-center gap-2">
                                {langchainVerdict === "THREAT" ? <AlertTriangle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                                AI Agent Final Verdict: {langchainVerdict}
                            </AlertTitle>
                            {langchainReasoning && <AlertDescription className="mt-2 italic">{langchainReasoning}</AlertDescription>}
                        </Alert>
                    )}
                </CardContent>
            </Card>

            {/* Detailed Analysis Accordion */}
            <Accordion type="multiple" defaultValue={["ml", "council", "investigations"]} className="space-y-4">

                {/* ML Analysis Section */}
                <AccordionItem value="ml" className="border rounded-lg px-4">
                    <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center gap-2">
                            <Activity className="h-5 w-5 text-blue-500" />
                            <span className="font-semibold">Machine Learning Analysis</span>
                            <Badge variant="outline">{mlConfidence}% confident</Badge>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent>
                        <div className="space-y-4 pt-4">
                            {/* Top Predictions */}
                            <div>
                                <h4 className="text-sm font-semibold mb-3">Threat Class Probabilities</h4>
                                <div className="space-y-2">
                                    {topPredictions.map((pred, idx) => (
                                        <div key={idx} className="space-y-1">
                                            <div className="flex justify-between text-sm">
                                                <span className="font-medium">{pred.name}</span>
                                                <span className="text-muted-foreground">{pred.probability.toFixed(1)}%</span>
                                            </div>
                                            <Progress value={pred.probability} className="h-2" />
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <Separator />

                            {/* Detection Metadata */}
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-muted-foreground">Detection Method:</span>
                                    <Badge variant="secondary" className="ml-2">
                                        {incident.triage_note?.detection_method || "Standard ML"}
                                    </Badge>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Feature Dimensions:</span>
                                    <span className="font-mono ml-2">
                                        {mlPrediction?.feature_count || 79}D
                                    </span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Uncertainty Score:</span>
                                    <span className="font-mono ml-2">
                                        {mlPrediction?.uncertainty_score ? `${(mlPrediction.uncertainty_score * 100).toFixed(1)}%` : "N/A"}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Event Count:</span>
                                    <span className="font-mono ml-2">{incident.detailed_events?.length || 0}</span>
                                </div>
                            </div>

                            {/* Gate Results (if multi-gate) */}
                            {incident.triage_note?.gate_results?.length > 0 && (
                                <>
                                    <Separator />
                                    <div>
                                        <h4 className="text-sm font-semibold mb-2">Detection Gates</h4>
                                        <div className="space-y-2">
                                            {incident.triage_note.gate_results.map((gate: any, idx: number) => (
                                                <div key={idx} className="flex items-center justify-between text-sm p-2 bg-muted/30 rounded">
                                                    <span className="font-medium">{gate.gate.replace(/_/g, " ")}</span>
                                                    <div className="flex items-center gap-2">
                                                        {gate.verdict === "pass" || gate.verdict === "escalate" ? (
                                                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                                                        ) : (
                                                            <XCircle className="h-4 w-4 text-red-500" />
                                                        )}
                                                        <Badge variant="outline" className="text-xs">{gate.verdict}</Badge>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </AccordionContent>
                </AccordionItem>

                {/* Council of Models Section */}
                <AccordionItem value="council" className="border rounded-lg px-4">
                    <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center gap-2">
                            <Shield className="h-5 w-5 text-purple-500" />
                            <span className="font-semibold">Council of Models Intelligence</span>
                            <Badge variant="outline">4 AI Models</Badge>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent>
                        <div className="grid md:grid-cols-2 gap-4 pt-4">
                            {/* Gemini */}
                            {geminiAnalysis && (
                                <Card className="border-blue-500/30">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2 text-blue-500">
                                            <Bot className="h-4 w-4" />
                                            Gemini Analysis
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="text-xs space-y-2">
                                        <p className="italic text-muted-foreground">"{geminiAnalysis.reasoning}"</p>
                                        <div className="flex justify-between">
                                            <span>Confidence:</span>
                                            <span className="font-mono">{geminiAnalysis.confidence || "N/A"}</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* Grok */}
                            {grokIntel && (
                                <Card className="border-purple-500/30">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2 text-purple-500">
                                            <Search className="h-4 w-4" />
                                            Grok Threat Intel
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="text-xs space-y-2">
                                        <div>
                                            <strong>Threat Actor:</strong> {grokIntel.threat_actor || "Unknown"}
                                        </div>
                                        <div>
                                            <strong>TTPs:</strong>
                                            <div className="flex flex-wrap gap-1 mt-1">
                                                {(grokIntel.ttps || []).map((ttp: string, i: number) => (
                                                    <Badge key={i} variant="outline" className="text-[10px]">{ttp}</Badge>
                                                ))}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>
                            )}

                            {/* OpenAI */}
                            {openaiRemediation && (
                                <Card className="border-green-500/30">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2 text-green-500">
                                            <Zap className="h-4 w-4" />
                                            OpenAI Remediation
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="text-xs">
                                        <ul className="space-y-1">
                                            {(openaiRemediation.recommended_actions || []).map((action: string, i: number) => (
                                                <li key={i} className="flex items-start gap-2">
                                                    <span className="text-primary mt-0.5">•</span>
                                                    <span>{action}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            )}

                            {/* LangChain Actions */}
                            {langchainActions.length > 0 && (
                                <Card className="border-orange-500/30">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="text-sm flex items-center gap-2 text-orange-500">
                                            <Target className="h-4 w-4" />
                                            LangChain Agent
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="text-xs">
                                        <ul className="space-y-1">
                                            {langchainActions.map((action: any, i: number) => (
                                                <li key={i} className="flex items-start gap-2">
                                                    <CheckCircle2 className="h-3 w-3 text-green-500 mt-0.5" />
                                                    <span>{typeof action === 'string' ? action : action.action || action.tool || JSON.stringify(action)}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            )}
                        </div>
                    </AccordionContent>
                </AccordionItem>

                {/* Investigation Results Section */}
                <AccordionItem value="investigations" className="border rounded-lg px-4">
                    <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center gap-2">
                            <FileSearch className="h-5 w-5 text-amber-500" />
                            <span className="font-semibold">Tool Execution & Investigation Results</span>
                            <div className="flex items-center gap-2">
                                <Badge variant="outline" className="animate-pulse border-green-500 text-green-500">Live</Badge>
                                <Badge variant="outline">{investigations.length} tools executed</Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent>
                        <div className="space-y-3 pt-4">
                            {isLoading ? (
                                <div className="flex items-center justify-center py-8">
                                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground mr-2" />
                                    <span className="text-muted-foreground">Loading investigations...</span>
                                </div>
                            ) : investigations.length === 0 ? (
                                <div className="text-center py-8 text-muted-foreground">
                                    <Eye className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                    <p>No tool executions recorded yet.</p>
                                    <p className="text-xs mt-1">Results will appear here when AI agents execute tools.</p>
                                </div>
                            ) : (
                                investigations.map((inv: any) => (
                                    <Card key={inv.id} className="border-l-4" style={{
                                        borderLeftColor: inv.status === "completed" ? "#10b981" :
                                            inv.status === "failed" ? "#ef4444" : "#6b7280"
                                    }}>
                                        <CardHeader className="pb-2">
                                            <div className="flex items-center justify-between">
                                                <CardTitle className="text-sm flex items-center gap-2">
                                                    <ListChecks className="h-4 w-4" />
                                                    {inv.tool_name}
                                                </CardTitle>
                                                <div className="flex items-center gap-2">
                                                    <Badge variant={inv.status === "completed" ? "default" : "destructive"}>
                                                        {inv.status}
                                                    </Badge>
                                                    {inv.severity && (
                                                        <Badge variant="outline">{inv.severity}</Badge>
                                                    )}
                                                </div>
                                            </div>
                                        </CardHeader>
                                        <CardContent className="text-xs space-y-2">
                                            {/* Execution Details */}
                                            <div className="grid grid-cols-2 gap-2 text-muted-foreground">
                                                <div>Category: <span className="text-foreground">{inv.tool_category}</span></div>
                                                <div>Execution Time: <span className="text-foreground font-mono">{inv.execution_time_ms}ms</span></div>
                                                <div>Confidence: <span className="text-foreground font-mono">{(inv.confidence_score * 100).toFixed(0)}%</span></div>
                                                <div>Findings: <span className="text-foreground font-mono">{inv.findings_count || 0}</span></div>
                                            </div>

                                            {/* Results Preview */}
                                            {inv.results && (
                                                <div className="mt-2">
                                                    <div className="text-muted-foreground mb-1">Results:</div>
                                                    <ScrollArea className="h-[60px] w-full rounded border bg-muted/30 p-2">
                                                        <pre className="text-[10px] font-mono whitespace-pre-wrap">
                                                            {typeof inv.results === 'string' ? inv.results : JSON.stringify(inv.results, null, 2)}
                                                        </pre>
                                                    </ScrollArea>
                                                </div>
                                            )}

                                            {/* IOCs */}
                                            {inv.iocs_discovered && inv.iocs_discovered.length > 0 && (
                                                <div className="mt-2">
                                                    <div className="text-muted-foreground mb-1">IOCs Discovered:</div>
                                                    <div className="flex flex-wrap gap-1">
                                                        {inv.iocs_discovered.map((ioc: string, i: number) => (
                                                            <Badge key={i} variant="outline" className="text-[9px] font-mono">
                                                                {ioc}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Error */}
                                            {inv.error_message && (
                                                <Alert variant="destructive" className="mt-2">
                                                    <AlertDescription className="text-xs">{inv.error_message}</AlertDescription>
                                                </Alert>
                                            )}
                                        </CardContent>
                                    </Card>
                                ))
                            )}
                        </div>
                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        </div>
    );
}
