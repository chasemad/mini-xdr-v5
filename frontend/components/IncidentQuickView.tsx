"use client";

import React, { useState, useEffect } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  AlertTriangle,
  Shield,
  Clock,
  MapPin,
  Activity,
  ExternalLink,
  Ban,
  Crosshair,
  Database,
  Zap,
  MessageSquare,
  Eye,
  TrendingUp
} from "lucide-react";
import Link from "next/link";
import AIChatInterface from "@/app/components/AIChatInterface";

interface IncidentQuickViewProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  incident: any;
  onExecuteAction?: (action: string) => void;
}

export function IncidentQuickView({
  open,
  onOpenChange,
  incident,
  onExecuteAction
}: IncidentQuickViewProps) {

  const [activeTab, setActiveTab] = useState("overview");

  // Reset to overview when incident changes
  useEffect(() => {
    if (open && incident) {
      setActiveTab("overview");
    }
  }, [open, incident?.id]);

  if (!incident) return null;

  const getSeverityColor = (severity?: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
      case 'high':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      default:
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'open':
      case 'new':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'investigating':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'contained':
        return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
      case 'resolved':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      default:
        return 'bg-gray-500/10 text-gray-500 border-gray-500/20';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const riskScore = incident.risk_score || 0;
  const riskPercentage = Math.round(riskScore * 100);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-2xl p-0 flex flex-col">
        <SheetHeader className="px-6 pt-6 pb-4 border-b border-border">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <SheetTitle className="text-xl">Incident #{incident.id}</SheetTitle>
                  <SheetDescription className="flex items-center gap-2 mt-1">
                    <MapPin className="w-3 h-3" />
                    <span className="font-mono text-xs">{incident.src_ip}</span>
                    <span className="text-xs">•</span>
                    <Clock className="w-3 h-3" />
                    <span className="text-xs">{formatDate(incident.created_at)}</span>
                  </SheetDescription>
                </div>
              </div>

              <div className="flex items-center gap-2 mt-3">
                <Badge className={getSeverityColor(incident.escalation_level || incident.triage_note?.severity)}>
                  {(incident.escalation_level || incident.triage_note?.severity || 'medium').toUpperCase()}
                </Badge>
                <Badge variant="outline" className={getStatusColor(incident.status)}>
                  {incident.status.toUpperCase()}
                </Badge>
                {incident.auto_contained && (
                  <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20">
                    AUTO-CONTAINED
                  </Badge>
                )}
              </div>
            </div>

            {/* Full Details Link */}
            <Button asChild variant="ghost" size="sm" className="gap-2">
              <Link href={`/incidents/incident/${incident.id}`}>
                <ExternalLink className="w-4 h-4" />
                <span className="hidden sm:inline">Full Details</span>
              </Link>
            </Button>
          </div>
        </SheetHeader>

        {/* Tabs Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
          <div className="px-6 pt-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview" className="text-xs sm:text-sm">
                <Activity className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="actions" className="text-xs sm:text-sm">
                <Zap className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
                Quick Actions
              </TabsTrigger>
              <TabsTrigger value="copilot" className="text-xs sm:text-sm">
                <MessageSquare className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
                Copilot
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-hidden">
            <TabsContent value="overview" className="h-full m-0 p-0">
              <ScrollArea className="h-full px-6 py-4">
                <div className="space-y-6">
                  {/* Threat Summary */}
                  <div>
                    <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                      <Shield className="w-4 h-4 text-primary" />
                      Threat Summary
                    </h3>
                    <div className="bg-surface-1 rounded-lg p-4 border border-border">
                      <p className="text-sm text-text-muted leading-relaxed">
                        {incident.reason || 'No threat description available'}
                      </p>
                      {incident.triage_note?.summary && (
                        <div className="mt-3 pt-3 border-t border-border">
                          <p className="text-xs font-medium text-text mb-1">AI Analysis:</p>
                          <p className="text-xs text-text-muted">{incident.triage_note.summary}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Risk Metrics */}
                  <div>
                    <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-primary" />
                      Risk Assessment
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-surface-1 rounded-lg p-4 border border-border">
                        <div className="text-xs text-text-muted mb-1">Risk Score</div>
                        <div className="flex items-baseline gap-2">
                          <span className="text-2xl font-bold text-text">{riskPercentage}%</span>
                          <span className={`text-xs ${riskPercentage > 70 ? 'text-red-500' : riskPercentage > 40 ? 'text-yellow-500' : 'text-green-500'}`}>
                            {riskPercentage > 70 ? 'High' : riskPercentage > 40 ? 'Medium' : 'Low'}
                          </span>
                        </div>
                      </div>

                      <div className="bg-surface-1 rounded-lg p-4 border border-border">
                        <div className="text-xs text-text-muted mb-1">ML Confidence</div>
                        <div className="flex items-baseline gap-2">
                          <span className="text-2xl font-bold text-text">
                            {Math.round((incident.agent_confidence || 0) * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Indicators of Compromise */}
                  {incident.iocs && (
                    <div>
                      <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                        <Crosshair className="w-4 h-4 text-primary" />
                        Indicators of Compromise
                      </h3>
                      <div className="space-y-2">
                        {incident.iocs.ip_addresses?.length > 0 && (
                          <div className="bg-surface-1 rounded-lg p-3 border border-border">
                            <div className="text-xs font-medium text-text mb-2">IP Addresses ({incident.iocs.ip_addresses.length})</div>
                            <div className="flex flex-wrap gap-1.5">
                              {incident.iocs.ip_addresses.slice(0, 5).map((ip: string, idx: number) => (
                                <span key={idx} className="px-2 py-1 bg-surface-0 border border-border rounded text-xs font-mono">
                                  {ip}
                                </span>
                              ))}
                              {incident.iocs.ip_addresses.length > 5 && (
                                <span className="px-2 py-1 text-xs text-text-muted">
                                  +{incident.iocs.ip_addresses.length - 5} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}

                        {incident.iocs.domains?.length > 0 && (
                          <div className="bg-surface-1 rounded-lg p-3 border border-border">
                            <div className="text-xs font-medium text-text mb-2">Domains ({incident.iocs.domains.length})</div>
                            <div className="flex flex-wrap gap-1.5">
                              {incident.iocs.domains.slice(0, 3).map((domain: string, idx: number) => (
                                <span key={idx} className="px-2 py-1 bg-surface-0 border border-border rounded text-xs font-mono">
                                  {domain}
                                </span>
                              ))}
                              {incident.iocs.domains.length > 3 && (
                                <span className="px-2 py-1 text-xs text-text-muted">
                                  +{incident.iocs.domains.length - 3} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Recent Events */}
                  {incident.detailed_events && incident.detailed_events.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-text mb-3 flex items-center gap-2">
                        <Database className="w-4 h-4 text-primary" />
                        Recent Events ({incident.detailed_events.length})
                      </h3>
                      <div className="space-y-2">
                        {incident.detailed_events.slice(0, 5).map((event: any, idx: number) => (
                          <div key={idx} className="bg-surface-1 rounded-lg p-3 border border-border text-xs">
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-mono text-primary">{event.eventid || 'Unknown'}</span>
                              <span className="text-text-muted">{formatDate(event.ts)}</span>
                            </div>
                            <p className="text-text-muted truncate">{event.message || 'No message'}</p>
                          </div>
                        ))}
                        {incident.detailed_events.length > 5 && (
                          <p className="text-xs text-text-muted text-center pt-2">
                            +{incident.detailed_events.length - 5} more events
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="actions" className="h-full m-0 p-0">
              <ScrollArea className="h-full px-6 py-4">
                <div className="space-y-3">
                  <div className="text-sm text-text-muted mb-4">
                    Execute immediate response actions for this incident
                  </div>

                  {/* Quick Action Buttons */}
                  <div className="grid grid-cols-2 gap-3">
                    <Button
                      variant="outline"
                      className="h-auto py-4 flex-col gap-2 hover:bg-red-500/10 hover:border-red-500/50"
                      onClick={() => onExecuteAction?.('block_ip')}
                    >
                      <Ban className="w-5 h-5 text-red-500" />
                      <div className="text-center">
                        <div className="font-semibold text-xs">Block IP</div>
                        <div className="text-[10px] text-text-muted">Immediate containment</div>
                      </div>
                    </Button>

                    <Button
                      variant="outline"
                      className="h-auto py-4 flex-col gap-2 hover:bg-orange-500/10 hover:border-orange-500/50"
                      onClick={() => onExecuteAction?.('isolate_host')}
                    >
                      <Shield className="w-5 h-5 text-orange-500" />
                      <div className="text-center">
                        <div className="font-semibold text-xs">Isolate Host</div>
                        <div className="text-[10px] text-text-muted">Network isolation</div>
                      </div>
                    </Button>

                    <Button
                      variant="outline"
                      className="h-auto py-4 flex-col gap-2 hover:bg-blue-500/10 hover:border-blue-500/50"
                      onClick={() => onExecuteAction?.('threat_intel')}
                    >
                      <Crosshair className="w-5 h-5 text-blue-500" />
                      <div className="text-center">
                        <div className="font-semibold text-xs">Threat Intel</div>
                        <div className="text-[10px] text-text-muted">IP reputation lookup</div>
                      </div>
                    </Button>

                    <Button
                      variant="outline"
                      className="h-auto py-4 flex-col gap-2 hover:bg-purple-500/10 hover:border-purple-500/50"
                      onClick={() => onExecuteAction?.('investigate')}
                    >
                      <Eye className="w-5 h-5 text-purple-500" />
                      <div className="text-center">
                        <div className="font-semibold text-xs">Investigate</div>
                        <div className="text-[10px] text-text-muted">Deep analysis</div>
                      </div>
                    </Button>
                  </div>

                  <Separator className="my-4" />

                  {/* Action History */}
                  <div>
                    <h4 className="text-xs font-semibold text-text mb-3">Recent Actions</h4>
                    {incident.actions && incident.actions.length > 0 ? (
                      <div className="space-y-2">
                        {incident.actions.slice(0, 5).map((action: any, idx: number) => (
                          <div key={idx} className="bg-surface-1 rounded-lg p-3 border border-border">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-text">{action.action || 'Unknown'}</span>
                              <Badge
                                variant="outline"
                                className={
                                  action.result?.toLowerCase() === 'success'
                                    ? 'bg-green-500/10 text-green-500 border-green-500/20'
                                    : 'bg-red-500/10 text-red-500 border-red-500/20'
                                }
                              >
                                {action.result || 'Unknown'}
                              </Badge>
                            </div>
                            {action.detail && (
                              <p className="text-xs text-text-muted truncate">{action.detail}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-text-muted text-center py-4">No actions taken yet</p>
                    )}
                  </div>
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="copilot" className="h-full m-0 p-0 flex flex-col">
              <div className="flex-1 overflow-hidden">
                <AIChatInterface selectedIncident={incident} />
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

export default IncidentQuickView;
