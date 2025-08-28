"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import Link from "next/link";

interface Investigation {
  id: string;
  title: string;
  description: string;
  status: "open" | "investigating" | "closed" | "escalated";
  priority: "low" | "medium" | "high" | "critical";
  assignee: string;
  created_at: string;
  updated_at: string;
  incident_ids: number[];
  tags: string[];
  evidence_count: number;
  timeline_events: number;
}

interface Evidence {
  id: string;
  investigation_id: string;
  type: "log" | "file" | "network" | "screenshot" | "note";
  title: string;
  content: string;
  hash?: string;
  source: string;
  collected_at: string;
  collected_by: string;
}

interface TimelineEvent {
  id: string;
  timestamp: string;
  title: string;
  description: string;
  source: string;
  type: "incident" | "evidence" | "action" | "note";
  confidence: number;
}

export default function InvestigationsPage() {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [selectedInvestigation, setSelectedInvestigation] = useState<Investigation | null>(null);
  const [activeTab, setActiveTab] = useState("list");
  const [newInvestigation, setNewInvestigation] = useState({
    title: "",
    description: "",
    priority: "medium" as const,
    assignee: "Current User"
  });
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [newEvidence, setNewEvidence] = useState({
    type: "note" as const,
    title: "",
    content: "",
    source: ""
  });

  // Load investigations and related data from backend
  useEffect(() => {
    const loadInvestigations = async () => {
      try {
        const response = await fetch('/api/investigations');
        if (response.ok) {
          const data = await response.json();
          setInvestigations(data.investigations || []);
        }
      } catch (error) {
        console.error('Failed to load investigations:', error);
      }
    };

    const loadEvidence = async () => {
      try {
        const response = await fetch('/api/investigations/evidence');
        if (response.ok) {
          const data = await response.json();
          setEvidence(data.evidence || []);
        }
      } catch (error) {
        console.error('Failed to load evidence:', error);
      }
    };

    const loadTimeline = async () => {
      try {
        const response = await fetch('/api/investigations/timeline');
        if (response.ok) {
          const data = await response.json();
          setTimeline(data.timeline || []);
        }
      } catch (error) {
        console.error('Failed to load timeline:', error);
      }
    };

    loadInvestigations();
    loadEvidence();
    loadTimeline();
  }, []);

  const createInvestigation = async () => {
    if (!newInvestigation.title.trim()) return;

    try {
      const response = await fetch('/api/investigations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: newInvestigation.title,
          description: newInvestigation.description,
          priority: newInvestigation.priority,
          assignee: newInvestigation.assignee
        }),
      });

      if (response.ok) {
        const investigation = await response.json();
        setInvestigations(prev => [...prev, investigation]);
        setNewInvestigation({ title: "", description: "", priority: "medium", assignee: "Current User" });
        setActiveTab("list");
      }
    } catch (error) {
      console.error('Failed to create investigation:', error);
    }
  };

  const addEvidence = async () => {
    if (!newEvidence.title.trim() || !selectedInvestigation) return;

    try {
      const response = await fetch('/api/investigations/evidence', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          investigation_id: selectedInvestigation.id,
          type: newEvidence.type,
          title: newEvidence.title,
          content: newEvidence.content,
          source: newEvidence.source || "Manual Entry"
        }),
      });

      if (response.ok) {
        const evidenceItem = await response.json();
        setEvidence(prev => [...prev, evidenceItem]);
        setNewEvidence({ type: "note", title: "", content: "", source: "" });
      }
    } catch (error) {
      console.error('Failed to add evidence:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "open": return "bg-blue-100 text-blue-800 border-blue-200";
      case "investigating": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "escalated": return "bg-red-100 text-red-800 border-red-200";
      case "closed": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical": return "bg-red-100 text-red-800 border-red-200";
      case "high": return "bg-orange-100 text-orange-800 border-orange-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getEvidenceIcon = (type: string) => {
    switch (type) {
      case "log": return "📄";
      case "file": return "📁";
      case "network": return "🌐";
      case "screenshot": return "📸";
      case "note": return "📝";
      default: return "📋";
    }
  };

  const getTimelineIcon = (type: string) => {
    switch (type) {
      case "incident": return "🚨";
      case "evidence": return "🔍";
      case "action": return "⚡";
      case "note": return "📝";
      default: return "📋";
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Investigations</h1>
          <p className="text-gray-600">Case management and collaborative investigation workspace</p>
        </div>
        <Button onClick={() => setActiveTab("create")}>
          ➕ New Investigation
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="list">All Investigations</TabsTrigger>
          <TabsTrigger value="details">Investigation Details</TabsTrigger>
          <TabsTrigger value="create">Create Investigation</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="list" className="space-y-6">
          <div className="grid gap-4">
            {investigations.map((investigation) => (
              <Card key={investigation.id} className="cursor-pointer hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-lg">{investigation.title}</h3>
                      <Badge className={`${getStatusColor(investigation.status)} border`}>
                        {investigation.status.replace("_", " ").toUpperCase()}
                      </Badge>
                      <Badge className={`${getPriorityColor(investigation.priority)} border`}>
                        {investigation.priority.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">ID: {investigation.id}</span>
                      <Button 
                        size="sm" 
                        onClick={() => {
                          setSelectedInvestigation(investigation);
                          setActiveTab("details");
                        }}
                      >
                        Open
                      </Button>
                    </div>
                  </div>

                  <p className="text-gray-700 mb-3">{investigation.description}</p>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Assignee:</span>
                      <p className="font-medium">{investigation.assignee}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Evidence:</span>
                      <p className="font-medium">{investigation.evidence_count} items</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Timeline:</span>
                      <p className="font-medium">{investigation.timeline_events} events</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Updated:</span>
                      <p className="font-medium">{new Date(investigation.updated_at).toLocaleDateString()}</p>
                    </div>
                  </div>

                  {investigation.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {investigation.tags.map((tag, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {investigation.incident_ids.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <span className="text-sm text-gray-600">Related Incidents: </span>
                      {investigation.incident_ids.map((id, index) => (
                        <Link key={id} href={`/incidents/${id}`}>
                          <Badge variant="outline" className="ml-1 cursor-pointer hover:bg-gray-100">
                            #{id}
                          </Badge>
                        </Link>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="details" className="space-y-6">
          {selectedInvestigation ? (
            <div className="space-y-6">
              {/* Investigation Header */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-3">
                        {selectedInvestigation.title}
                        <Badge className={`${getStatusColor(selectedInvestigation.status)} border`}>
                          {selectedInvestigation.status.toUpperCase()}
                        </Badge>
                      </CardTitle>
                      <p className="text-gray-600 mt-1">{selectedInvestigation.description}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">{selectedInvestigation.id}</p>
                      <p className="text-sm text-gray-600">
                        Created: {new Date(selectedInvestigation.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </CardHeader>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Evidence Collection */}
                <Card>
                  <CardHeader>
                    <CardTitle>Evidence ({evidence.filter(e => e.investigation_id === selectedInvestigation.id).length})</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Add Evidence Form */}
                    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                      <h4 className="font-medium mb-2">Add Evidence</h4>
                      <div className="space-y-2">
                        <div className="flex gap-2">
                          <Select value={newEvidence.type} onValueChange={(value: any) => setNewEvidence({...newEvidence, type: value})}>
                            <SelectTrigger className="w-32">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="log">Log</SelectItem>
                              <SelectItem value="file">File</SelectItem>
                              <SelectItem value="network">Network</SelectItem>
                              <SelectItem value="screenshot">Screenshot</SelectItem>
                              <SelectItem value="note">Note</SelectItem>
                            </SelectContent>
                          </Select>
                          <Input
                            placeholder="Evidence title"
                            value={newEvidence.title}
                            onChange={(e) => setNewEvidence({...newEvidence, title: e.target.value})}
                            className="flex-1"
                          />
                        </div>
                        <textarea
                          placeholder="Evidence description or content..."
                          value={newEvidence.content}
                          onChange={(e) => setNewEvidence({...newEvidence, content: e.target.value})}
                          className="w-full h-20 p-2 border border-gray-300 rounded text-sm"
                        />
                        <div className="flex gap-2">
                          <Input
                            placeholder="Source"
                            value={newEvidence.source}
                            onChange={(e) => setNewEvidence({...newEvidence, source: e.target.value})}
                            className="flex-1"
                          />
                          <Button size="sm" onClick={addEvidence} disabled={!newEvidence.title.trim()}>
                            Add
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Evidence List */}
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {evidence
                        .filter(e => e.investigation_id === selectedInvestigation.id)
                        .map((item) => (
                          <div key={item.id} className="border border-gray-200 rounded-lg p-3">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-lg">{getEvidenceIcon(item.type)}</span>
                              <span className="font-medium">{item.title}</span>
                              <Badge variant="outline" className="text-xs">{item.type}</Badge>
                            </div>
                            <p className="text-sm text-gray-700 mb-2">{item.content}</p>
                            <div className="flex justify-between text-xs text-gray-500">
                              <span>Source: {item.source}</span>
                              <span>By: {item.collected_by}</span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Timeline */}
                <Card>
                  <CardHeader>
                    <CardTitle>Investigation Timeline</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {timeline.map((event) => (
                        <div key={event.id} className="flex gap-3">
                          <div className="flex flex-col items-center">
                            <span className="text-lg">{getTimelineIcon(event.type)}</span>
                            <div className="w-px h-8 bg-gray-300 mt-1"></div>
                          </div>
                          <div className="flex-1 pb-4">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium">{event.title}</span>
                              <Badge variant="outline" className="text-xs">{event.type}</Badge>
                              <span className="text-xs text-gray-500">
                                {(event.confidence * 100).toFixed(0)}% confidence
                              </span>
                            </div>
                            <p className="text-sm text-gray-700 mb-1">{event.description}</p>
                            <div className="flex justify-between text-xs text-gray-500">
                              <span>{event.source}</span>
                              <span>{new Date(event.timestamp).toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          ) : (
            <Alert>
              <AlertDescription>
                Select an investigation from the list to view details.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create New Investigation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Investigation Title</Label>
                <Input
                  value={newInvestigation.title}
                  onChange={(e) => setNewInvestigation({...newInvestigation, title: e.target.value})}
                  placeholder="e.g., Suspected APT Activity - Network Reconnaissance"
                />
              </div>

              <div className="space-y-2">
                <Label>Description</Label>
                <textarea
                  value={newInvestigation.description}
                  onChange={(e) => setNewInvestigation({...newInvestigation, description: e.target.value})}
                  placeholder="Detailed description of what needs to be investigated..."
                  className="w-full h-24 p-3 border border-gray-300 rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Priority</Label>
                  <Select 
                    value={newInvestigation.priority} 
                    onValueChange={(value: any) => setNewInvestigation({...newInvestigation, priority: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Assignee</Label>
                  <Input
                    value={newInvestigation.assignee}
                    onChange={(e) => setNewInvestigation({...newInvestigation, assignee: e.target.value})}
                    placeholder="Assigned investigator"
                  />
                </div>
              </div>

              <div className="flex gap-2 pt-4">
                <Button onClick={createInvestigation} disabled={!newInvestigation.title.trim()}>
                  Create Investigation
                </Button>
                <Button variant="outline" onClick={() => setActiveTab("list")}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Investigation Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Active</span>
                    <span className="font-semibold">{investigations.filter(i => i.status !== "closed").length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Critical Priority</span>
                    <span className="font-semibold text-red-600">{investigations.filter(i => i.priority === "critical").length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Resolution</span>
                    <span className="font-semibold">4.2 days</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Evidence Items</span>
                    <span className="font-semibold">{investigations.reduce((sum, i) => sum + i.evidence_count, 0)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Status Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {["open", "investigating", "escalated", "closed"].map(status => {
                    const count = investigations.filter(i => i.status === status).length;
                    return (
                      <div key={status} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{status}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-200 rounded">
                            <div 
                              className={`h-full rounded ${getStatusColor(status).split(' ')[0]}`}
                              style={{ width: `${(count / investigations.length) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium">{count}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Team Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Sarah Chen</span>
                    <span className="font-medium">3 active</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Mike Rodriguez</span>
                    <span className="font-medium">2 active</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Alex Kim</span>
                    <span className="font-medium">1 active</span>
                  </div>
                  <div className="pt-2 border-t">
                    <div className="flex justify-between font-medium">
                      <span>Team Total</span>
                      <span>{investigations.filter(i => i.status !== "closed").length} active</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
