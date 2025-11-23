"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Shield, Search, FileText, Ghost, Activity, Radio, Brain, Terminal, Loader2 } from "lucide-react";
import { agentApi } from "@/lib/agent-api";

interface Message {
  role: "user" | "agent";
  content: string;
  timestamp: Date;
  confidence?: number;
  actions?: Array<{action: string; status: string}>;
}

interface AgentStatus {
  id: string;
  name: string;
  status: "online" | "offline" | "busy";
  lastActivity: Date;
  description: string;
}

interface MLModelStatus {
  enhanced_detector?: {
    loaded: boolean;
    status: string;
    model_type: string;
    device?: string;
  };
  federated_detector?: any;
}

export default function AgentsPage() {
  const [conversation, setConversation] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("containment");
  const [loading, setLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([
    {
      id: "containment",
      name: "Containment Orchestrator",
      status: "offline",
      lastActivity: new Date(),
      description: "Autonomous threat containment"
    },
    {
      id: "attribution",
      name: "Attribution Agent",
      status: "offline",
      lastActivity: new Date(),
      description: "Actor identification & analysis"
    },
    {
      id: "forensics",
      name: "Forensics Agent",
      status: "offline",
      lastActivity: new Date(),
      description: "Evidence collection & chain of custody"
    },
    {
      id: "deception",
      name: "Deception Agent",
      status: "offline",
      lastActivity: new Date(),
      description: "Honeypots & adversary engagement"
    }
  ]);
  const [mlStatus, setMlStatus] = useState<MLModelStatus>({});
  const [systemStatus, setSystemStatus] = useState<string>("checking");
  const [isLoading, setIsLoading] = useState(true);

  // Scroll to bottom of chat
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [conversation, loading]);

  // Fetch agent and ML model status on mount and periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        // Phase 2 Integration: Use Agent API
        const hubStatus = await agentApi.getHubStatus();
        setSystemStatus(hubStatus.system_health || "healthy");

        const agentsList = ["containment", "attribution", "forensics", "deception"];
        const agentStatusesPromises = agentsList.map(name => agentApi.getAgentStatus(name).catch(e => null));
        const agentsData = await Promise.all(agentStatusesPromises);

        const updatedStatuses: AgentStatus[] = agentsData.map((agent, index) => {
            if (!agent) {
                // Fallback if agent fetch fails
                return agentStatuses[index];
            }
            return {
                id: agent.agent_name,
                name: formatAgentName(agent.agent_name),
                status: agent.status === "operational" ? "online" : "offline",
                lastActivity: new Date(agent.last_active_timestamp),
                description: getAgentDescription(agent.agent_name)
            };
        });

        setAgentStatuses(updatedStatuses);

        // Mock ML Status for now or fetch from dedicated endpoint if needed
        setMlStatus({
            enhanced_detector: {
                loaded: true,
                status: "active",
                model_type: "Ensemble (RF+GBM+NN)",
                device: "CPU"
            }
        });

      } catch (error) {
        console.error('Failed to fetch agent status:', error);
        setSystemStatus("error");
      } finally {
        setIsLoading(false);
      }
    };

    // Fetch immediately
    fetchStatus();

    // Poll every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const formatAgentName = (id: string) => {
      const map: Record<string, string> = {
          containment: "Containment Orchestrator",
          attribution: "Attribution Agent",
          forensics: "Forensics Agent",
          deception: "Deception Agent"
      };
      return map[id] || id.charAt(0).toUpperCase() + id.slice(1);
  };

  const getAgentDescription = (id: string) => {
      const map: Record<string, string> = {
          containment: "Autonomous threat containment",
          attribution: "Actor identification & analysis",
          forensics: "Evidence collection & chain of custody",
          deception: "Honeypots & adversary engagement"
      };
      return map[id] || "AI Security Agent";
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date()
    };

    setConversation(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch('/api/agents/orchestrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_type: selectedAgent,
          query: input,
          history: conversation.slice(-5) // Last 5 messages for context
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const agentMessage: Message = {
        role: "agent",
        content: data.message || "No response",
        timestamp: new Date(),
        confidence: data.confidence,
        actions: data.actions
      };

      setConversation(prev => [...prev, agentMessage]);

    } catch (error) {
      console.error('Agent communication error:', error);
      const errorMessage: Message = {
        role: "agent",
        content: `Error: Failed to communicate with agent. ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearConversation = () => {
    setConversation([]);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'busy': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-slate-500';
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const agentCapabilities = [
    {
      id: "containment",
      title: "Containment",
      icon: Shield,
      color: "text-blue-500",
      capabilities: ["Incident Evaluation", "Blocking Actions", "Risk Scoring", "Policy Enforcement"]
    },
    {
      id: "attribution",
      title: "Attribution",
      icon: Search,
      color: "text-purple-500",
      capabilities: ["Actor ID", "Campaign Correlation", "TTP Analysis", "Confidence Scoring"]
    },
    {
      id: "forensics",
      title: "Forensics",
      icon: FileText,
      color: "text-amber-500",
      capabilities: ["Evidence Collection", "Chain of Custody", "Artifact Analysis", "Timeline Recon"]
    },
    {
      id: "deception",
      title: "Deception",
      icon: Ghost,
      color: "text-rose-500",
      capabilities: ["Honeypot Mgmt", "Attacker Profiling", "Deception Ops", "Intel Gathering"]
    }
  ];

  return (
    <DashboardLayout breadcrumbs={[{ label: "AI Agents" }]}>
      <div className="space-y-8 max-w-[1600px] mx-auto">
        {/* Header Section */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Agent Command Center</h2>
            <p className="text-muted-foreground mt-2">
              Interact with and monitor autonomous security agents powered by local ML models.
            </p>

            <div className="flex items-center gap-4 mt-4">
              {systemStatus === "healthy" && mlStatus.enhanced_detector?.loaded ? (
                <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20 px-3 py-1">
                  <Brain className="w-3 h-3 mr-2" />
                  Local Models Active • {mlStatus.enhanced_detector.model_type}
                </Badge>
              ) : (
                <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20 px-3 py-1">
                  <Loader2 className="w-3 h-3 mr-2 animate-spin" />
                  Initializing Models...
                </Badge>
              )}
            </div>
          </div>

          <div className="flex gap-2">
             <Button onClick={clearConversation} variant="outline" size="sm">
              <Terminal className="w-4 h-4 mr-2" />
              Clear Console
            </Button>
          </div>
        </div>

        {/* Agent Capabilities Grid (Moved to Top) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {agentCapabilities.map((agent) => {
            const Icon = agent.icon;
            const status = agentStatuses.find(a => a.id === agent.id);
            const isOnline = status?.status === 'online';

            return (
              <Card key={agent.id} className="bg-card transition-all duration-300 ease-in-out hover:-translate-y-1 hover:shadow-lg hover:shadow-primary/20 border-muted">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between mb-2">
                    <div className={`p-2 rounded-lg bg-background border ${agent.color}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className={`flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded-full ${
                      isOnline ? 'bg-green-500/10 text-green-500' : 'bg-slate-500/10 text-slate-500'
                    }`}>
                      <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-green-500' : 'bg-slate-500'}`} />
                      {isOnline ? 'ONLINE' : 'OFFLINE'}
                    </div>
                  </div>
                  <CardTitle className="text-lg">{agent.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {agent.capabilities.map((cap, i) => (
                      <li key={i} className="text-xs text-muted-foreground flex items-center gap-2">
                        <div className="w-1 h-1 rounded-full bg-primary/50" />
                        {cap}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[600px]">
          {/* Agent Status Panel - Compact Side */}
          <Card className="lg:col-span-1 flex flex-col h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="w-4 h-4 text-muted-foreground" />
                <span>System Telemetry</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto space-y-4">
              {isLoading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="p-3 border rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-2 w-2 rounded-full" />
                    </div>
                    <Skeleton className="h-3 w-full" />
                  </div>
                ))
              ) : (
                agentStatuses.map((agent) => (
                  <div key={agent.id} className="p-3 bg-muted/30 rounded-lg border border-border/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm">{agent.name}</span>
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)} shadow-[0_0_8px_currentColor]`} />
                    </div>
                    <p className="text-xs text-muted-foreground mb-3 leading-relaxed">{agent.description}</p>
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground/70 border-t border-border/50 pt-2">
                      <span>Latency: 12ms</span>
                      <span>{formatTimestamp(agent.lastActivity)}</span>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          {/* Main Chat Interface */}
          <Card className="lg:col-span-3 flex flex-col h-full border-primary/20 shadow-lg shadow-primary/5">
            <CardHeader className="border-b border-border/50 bg-muted/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Radio className="w-4 h-4 text-primary animate-pulse" />
                  <CardTitle>Live Operation Channel</CardTitle>
                </div>
                <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                  <SelectTrigger className="w-[240px] bg-background">
                    <SelectValue placeholder="Select agent channel" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="containment">Containment Orchestrator</SelectItem>
                    <SelectItem value="attribution">Attribution Agent</SelectItem>
                    <SelectItem value="forensics">Forensics Agent</SelectItem>
                    <SelectItem value="deception">Deception Agent</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>

            <CardContent className="flex-1 flex flex-col p-0 min-h-0">
              {/* Chat Messages */}
              <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
                {conversation.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50 space-y-4">
                    <Brain className="w-16 h-16 text-muted-foreground/20" />
                    <div className="text-center">
                      <p className="font-medium">Ready for instructions</p>
                      <p className="text-sm mt-1">Select an agent and initiate command sequence</p>
                    </div>
                    <div className="flex gap-2 text-xs mt-4">
                      <span className="bg-muted px-2 py-1 rounded">Evaluate incident 123</span>
                      <span className="bg-muted px-2 py-1 rounded">Analyze IP 8.8.8.8</span>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {conversation.map((message, idx) => (
                      <div key={idx} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] rounded-lg p-4 ${
                          message.role === 'user'
                            ? 'bg-primary text-primary-foreground ml-12'
                            : 'bg-muted mr-12 border border-border'
                        }`}>
                          <div className="flex items-center gap-2 mb-2 opacity-80 text-xs uppercase tracking-wider font-semibold">
                            <span>{message.role === 'user' ? 'Operator' : 'System'}</span>
                            <span>•</span>
                            <span>{formatTimestamp(message.timestamp)}</span>
                            {message.confidence && (
                              <>
                                <span>•</span>
                                <Badge variant="secondary" className="text-[10px] h-5">
                                  {Math.round(message.confidence * 100)}% CONFIDENCE
                                </Badge>
                              </>
                            )}
                          </div>

                          <div className="prose prose-sm dark:prose-invert whitespace-pre-wrap">
                            {message.content}
                          </div>

                          {/* Display actions if present */}
                          {message.actions && message.actions.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-border/50">
                              <p className="text-xs font-semibold mb-2 uppercase tracking-wide opacity-70">Executed Protocols:</p>
                              <div className="space-y-2">
                                {message.actions.map((action, actionIdx) => (
                                  <div key={actionIdx} className="text-sm bg-background/50 p-2 rounded border border-border/50 flex items-center justify-between">
                                    <span className="font-mono text-xs">{action.action}</span>
                                    <Badge variant={action.status === 'success' ? 'default' : 'destructive'} className="text-[10px] uppercase">
                                      {action.status}
                                    </Badge>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {loading && (
                      <div className="flex justify-start">
                        <div className="bg-muted p-4 rounded-lg rounded-tl-none max-w-[80%] flex items-center gap-3">
                          <Loader2 className="w-4 h-4 animate-spin text-primary" />
                          <span className="text-sm text-muted-foreground">Processing request...</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </ScrollArea>

              {/* Input Area */}
              <div className="p-4 border-t border-border bg-card">
                <div className="flex gap-3">
                  <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Enter command or query..."
                    className="flex-1 bg-background/50 font-mono text-sm"
                    disabled={loading}
                    autoFocus
                  />
                  <Button
                    onClick={sendMessage}
                    disabled={!input.trim() || loading}
                    className="px-6 shadow-lg shadow-primary/20"
                  >
                    Send
                  </Button>
                </div>

                {/* Quick Actions */}
                <div className="flex flex-wrap gap-2 mt-4">
                  <Button variant="ghost" size="sm" onClick={() => setInput("Show system status")} className="text-xs text-muted-foreground hover:text-primary">
                    Status
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setInput("List recent high-risk incidents")} className="text-xs text-muted-foreground hover:text-primary">
                    Incidents
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setInput("Run threat hunting scan")} className="text-xs text-muted-foreground hover:text-primary">
                    Threat Hunt
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}
