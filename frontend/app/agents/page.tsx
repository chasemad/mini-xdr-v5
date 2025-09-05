"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";

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

export default function AgentsPage() {
  const [conversation, setConversation] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("containment");
  const [loading, setLoading] = useState(false);
  const [agentStatuses] = useState<AgentStatus[]>([
    {
      id: "containment",
      name: "Containment Orchestrator",
      status: "online",
      lastActivity: new Date(),
      description: "AI agent for autonomous threat containment decisions"
    },
    {
      id: "hunter",
      name: "Threat Hunter",
      status: "online",
      lastActivity: new Date(),
      description: "Proactive threat hunting and investigation"
    },
    {
      id: "rollback",
      name: "Rollback Agent",
      status: "online", 
      lastActivity: new Date(),
      description: "Evaluates and reverses false positive actions"
    }
  ]);

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
      default: return 'bg-gray-500';
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString();
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Agents Control Panel</h1>
          <p className="text-gray-600">Interact with autonomous security agents</p>
        </div>
        <Button onClick={clearConversation} variant="outline">
          Clear Chat
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Agent Status Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>Agents Status</span>
              <Badge variant="outline">{agentStatuses.filter(a => a.status === 'online').length} Online</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {agentStatuses.map((agent) => (
              <div key={agent.id} className="p-3 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">{agent.name}</span>
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(agent.status)}`} />
                </div>
                <p className="text-xs text-gray-600 mb-2">{agent.description}</p>
                <p className="text-xs text-gray-500">
                  Last activity: {formatTimestamp(agent.lastActivity)}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Main Chat Interface */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Agent Interaction</CardTitle>
              <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Select agent" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="containment">Containment Orchestrator</SelectItem>
                  <SelectItem value="hunter">Threat Hunter</SelectItem>
                  <SelectItem value="rollback">Rollback Agent</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {/* Chat Messages */}
            <ScrollArea className="h-96 border rounded-lg p-4 mb-4">
              {conversation.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <p>Start a conversation with an AI agent</p>
                  <p className="text-sm mt-2">Try: &quot;Evaluate incident 123&quot; or &quot;Analyze IP 192.168.1.100&quot;</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {conversation.map((message, idx) => (
                    <div key={idx} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[70%] p-3 rounded-lg ${
                        message.role === 'user' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-gray-100 text-gray-900'
                      }`}>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="font-medium text-sm">
                            {message.role === 'user' ? 'You' : 'Agent'}
                          </span>
                          <span className="text-xs opacity-70">
                            {formatTimestamp(message.timestamp)}
                          </span>
                          {message.confidence && (
                            <Badge variant="outline" className="text-xs">
                              {Math.round(message.confidence * 100)}% confidence
                            </Badge>
                          )}
                        </div>
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        
                        {/* Display actions if present */}
                        {message.actions && message.actions.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-opacity-20">
                            <p className="text-sm font-medium mb-2">Actions Taken:</p>
                            <div className="space-y-1">
                              {message.actions.map((action, actionIdx) => (
                                <div key={actionIdx} className="text-sm bg-black bg-opacity-10 p-2 rounded">
                                  <span className="font-medium">{action.action}:</span> {action.status}
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
                      <div className="bg-gray-100 p-3 rounded-lg max-w-[70%]">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-900"></div>
                          <span>Agent is thinking...</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>

            {/* Input Area */}
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask agent: e.g., 'Evaluate IP 8.8.8.8 for containment' or 'Show recent incidents'"
                className="flex-1"
                disabled={loading}
              />
              <Button 
                onClick={sendMessage}
                disabled={!input.trim() || loading}
                className="px-6"
              >
                Send
              </Button>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 mt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("Show system status")}
              >
                System Status
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("List recent high-risk incidents")}
              >
                Recent Incidents
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("What are the current threat levels?")}
              >
                Threat Levels
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("Run threat hunting scan")}
              >
                Threat Hunt
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Help Section */}
      <Card>
        <CardHeader>
          <CardTitle>Agent Capabilities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Containment Orchestrator</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Evaluate incidents for containment</li>
                <li>• Execute blocking actions</li>
                <li>• Risk assessment and scoring</li>
                <li>• Policy-based decisions</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Threat Hunter</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Proactive threat discovery</li>
                <li>• Behavioral analysis</li>
                <li>• IOC correlation</li>
                <li>• Attack pattern detection</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Rollback Agent</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• False positive detection</li>
                <li>• Action reversal</li>
                <li>• Impact assessment</li>
                <li>• Learning from mistakes</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
