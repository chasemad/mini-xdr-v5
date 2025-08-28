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

interface HuntResult {
  id: string;
  timestamp: string;
  source_ip: string;
  event_type: string;
  description: string;
  risk_score: number;
  matches: Record<string, any>;
}

interface HuntQuery {
  name: string;
  query: string;
  description: string;
  risk_level: "low" | "medium" | "high" | "critical";
  last_run?: string;
  results_count?: number;
}

interface IOC {
  type: "ip" | "domain" | "hash" | "user_agent";
  value: string;
  confidence: number;
  source: string;
  first_seen: string;
  last_seen: string;
  threat_actor?: string;
}

export default function ThreatHuntingPage() {
  const [activeTab, setActiveTab] = useState("hunt");
  const [huntQuery, setHuntQuery] = useState("");
  const [huntResults, setHuntResults] = useState<HuntResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [savedQueries, setSavedQueries] = useState<HuntQuery[]>([]);
  const [queryName, setQueryName] = useState("");
  const [queryDescription, setQueryDescription] = useState("");
  const [iocList, setIocList] = useState<IOC[]>([]);
  const [newIOC, setNewIOC] = useState({ type: "ip", value: "", source: "manual" });

  // Load saved queries and IOCs from backend
  useEffect(() => {
    const loadSavedQueries = async () => {
      try {
        const response = await fetch('/api/hunt/queries');
        if (response.ok) {
          const data = await response.json();
          setSavedQueries(data.queries || []);
        }
      } catch (error) {
        console.error('Failed to load saved queries:', error);
      }
    };

    const loadIOCs = async () => {
      try {
        const response = await fetch('/api/intelligence/iocs');
        if (response.ok) {
          const data = await response.json();
          setIocList(data.iocs || []);
        }
      } catch (error) {
        console.error('Failed to load IOCs:', error);
      }
    };

    loadSavedQueries();
    loadIOCs();
  }, []);

  const runHunt = async () => {
    if (!huntQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/hunt/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: huntQuery,
          limit: 100
        }),
      });

      if (!response.ok) {
        throw new Error(`Hunt execution failed: ${response.status}`);
      }

      const data = await response.json();
      setHuntResults(data.results || []);
    } catch (error) {
      console.error("Hunt execution failed:", error);
      setHuntResults([]);
    } finally {
      setLoading(false);
    }
  };

  const saveQuery = async () => {
    if (!queryName.trim() || !huntQuery.trim()) return;
    
    try {
      const response = await fetch('/api/hunt/queries', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: queryName,
          query: huntQuery,
          description: queryDescription,
          risk_level: "medium"
        }),
      });

      if (response.ok) {
        const newQuery = await response.json();
        setSavedQueries(prev => [...prev, newQuery]);
        setQueryName("");
        setQueryDescription("");
      }
    } catch (error) {
      console.error('Failed to save query:', error);
    }
  };

  const addIOC = async () => {
    if (!newIOC.value.trim()) return;
    
    try {
      const response = await fetch('/api/intelligence/iocs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: newIOC.type,
          value: newIOC.value,
          source: newIOC.source,
          confidence: 0.8
        }),
      });

      if (response.ok) {
        const newIOCData = await response.json();
        setIocList(prev => [...prev, newIOCData]);
        setNewIOC({ type: "ip", value: "", source: "manual" });
      }
    } catch (error) {
      console.error('Failed to add IOC:', error);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "critical": return "bg-red-100 text-red-800 border-red-200";
      case "high": return "bg-orange-100 text-orange-800 border-orange-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getRiskScoreColor = (score: number) => {
    if (score >= 0.8) return "text-red-600";
    if (score >= 0.6) return "text-orange-600";
    if (score >= 0.4) return "text-yellow-600";
    return "text-green-600";
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Threat Hunting</h1>
          <p className="text-gray-600">Proactive threat detection and investigation</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 bg-green-500 rounded-full"></div>
          <span className="text-sm text-gray-600">Hunt Engine Ready</span>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="hunt">Interactive Hunt</TabsTrigger>
          <TabsTrigger value="queries">Saved Queries</TabsTrigger>
          <TabsTrigger value="iocs">IOC Management</TabsTrigger>
          <TabsTrigger value="analytics">Hunt Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="hunt" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Query Builder */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Hunt Query Builder</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Query</Label>
                    <textarea
                      value={huntQuery}
                      onChange={(e) => setHuntQuery(e.target.value)}
                      placeholder="Enter hunt query (e.g., eventid:cowrie.login.failed AND src_ip:203.0.113.*)"
                      className="w-full h-32 p-3 border border-gray-300 rounded-lg text-sm font-mono"
                    />
                  </div>
                  
                  <div className="flex gap-2">
                    <Button onClick={runHunt} disabled={loading || !huntQuery.trim()}>
                      {loading ? "Hunting..." : "🔍 Run Hunt"}
                    </Button>
                    
                    <Button variant="outline" onClick={() => setHuntQuery("")}>
                      Clear
                    </Button>
                  </div>

                  {/* Quick Queries */}
                  <div className="border-t pt-4">
                    <Label className="text-sm font-medium">Quick Hunt Templates:</Label>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("eventid:cowrie.login.failed | count by src_ip | where count > 5")}
                      >
                        SSH Brute Force
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("src_ip:192.168.* AND dst_ip:10.0.0.*")}
                      >
                        Lateral Movement
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("user_agent:/(curl|wget|python)/")}
                      >
                        Suspicious Agents
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("timestamp:[22:00 TO 06:00]")}
                      >
                        Off-Hours Activity
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Save Query Panel */}
            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Save Query</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Query Name</Label>
                    <Input
                      value={queryName}
                      onChange={(e) => setQueryName(e.target.value)}
                      placeholder="e.g., Advanced SSH Hunt"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <textarea
                      value={queryDescription}
                      onChange={(e) => setQueryDescription(e.target.value)}
                      placeholder="Describe what this hunt detects..."
                      className="w-full h-20 p-2 border border-gray-300 rounded text-sm"
                    />
                  </div>
                  
                  <Button 
                    onClick={saveQuery} 
                    disabled={!queryName.trim() || !huntQuery.trim()}
                    className="w-full"
                  >
                    💾 Save Query
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Hunt Results */}
          {huntResults.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Hunt Results ({huntResults.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {huntResults.map((result) => (
                    <div key={result.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">
                            {result.source_ip}
                          </span>
                          <Badge variant="outline">{result.event_type}</Badge>
                          <span className={`font-semibold ${getRiskScoreColor(result.risk_score)}`}>
                            Risk: {(result.risk_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {new Date(result.timestamp).toLocaleString()}
                        </span>
                      </div>
                      
                      <p className="text-gray-700 mb-2">{result.description}</p>
                      
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <strong>Matches:</strong> {JSON.stringify(result.matches)}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="queries" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Saved Hunt Queries</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {savedQueries.map((query, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">{query.name}</h3>
                      <div className="flex items-center gap-2">
                        <Badge className={`${getRiskColor(query.risk_level)} border`}>
                          {query.risk_level.toUpperCase()}
                        </Badge>
                        <Button 
                          size="sm" 
                          onClick={() => setHuntQuery(query.query)}
                        >
                          Load
                        </Button>
                      </div>
                    </div>
                    
                    <p className="text-gray-600 text-sm mb-2">{query.description}</p>
                    
                    <div className="bg-gray-50 p-2 rounded text-xs font-mono mb-2">
                      {query.query}
                    </div>
                    
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span>Last run: {query.last_run ? new Date(query.last_run).toLocaleString() : "Never"}</span>
                      <span>Results: {query.results_count || 0}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="iocs" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Indicators of Compromise (IOCs)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {iocList.map((ioc, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">{ioc.type.toUpperCase()}</Badge>
                            <span className="font-mono text-sm">{ioc.value}</span>
                            <span className="text-sm text-gray-600">
                              Confidence: {(ioc.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <span className="text-xs text-gray-500">{ioc.source}</span>
                        </div>
                        
                        {ioc.threat_actor && (
                          <p className="text-sm text-red-600 font-medium">
                            Threat Actor: {ioc.threat_actor}
                          </p>
                        )}
                        
                        <div className="flex gap-4 text-xs text-gray-500 mt-1">
                          <span>First seen: {new Date(ioc.first_seen).toLocaleDateString()}</span>
                          <span>Last seen: {new Date(ioc.last_seen).toLocaleDateString()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Add IOC</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Type</Label>
                    <Select 
                      value={newIOC.type} 
                      onValueChange={(value) => setNewIOC({...newIOC, type: value})}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ip">IP Address</SelectItem>
                        <SelectItem value="domain">Domain</SelectItem>
                        <SelectItem value="hash">File Hash</SelectItem>
                        <SelectItem value="user_agent">User Agent</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Value</Label>
                    <Input
                      value={newIOC.value}
                      onChange={(e) => setNewIOC({...newIOC, value: e.target.value})}
                      placeholder="Enter IOC value..."
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Source</Label>
                    <Input
                      value={newIOC.source}
                      onChange={(e) => setNewIOC({...newIOC, source: e.target.value})}
                      placeholder="e.g., VirusTotal, Manual"
                    />
                  </div>
                  
                  <Button onClick={addIOC} disabled={!newIOC.value.trim()} className="w-full">
                    ➕ Add IOC
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Hunt Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Hunts</span>
                    <span className="font-semibold">47</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">This Week</span>
                    <span className="font-semibold">12</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Results</span>
                    <span className="font-semibold">8.3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Success Rate</span>
                    <span className="font-semibold text-green-600">85.1%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Threats Found</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">SSH Brute Force</span>
                    <Badge className="bg-red-100 text-red-800">15</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Port Scanning</span>
                    <Badge className="bg-orange-100 text-orange-800">8</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Lateral Movement</span>
                    <Badge className="bg-yellow-100 text-yellow-800">3</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Suspicious Agents</span>
                    <Badge className="bg-blue-100 text-blue-800">12</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="border-l-2 border-green-500 pl-2">
                    <p className="font-medium">SSH Hunt completed</p>
                    <p className="text-gray-500 text-xs">5 minutes ago</p>
                  </div>
                  <div className="border-l-2 border-blue-500 pl-2">
                    <p className="font-medium">New IOC added</p>
                    <p className="text-gray-500 text-xs">12 minutes ago</p>
                  </div>
                  <div className="border-l-2 border-orange-500 pl-2">
                    <p className="font-medium">High-risk findings</p>
                    <p className="text-gray-500 text-xs">1 hour ago</p>
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
