"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Target, Search, Globe, AlertTriangle, Shield, Workflow, Activity, BarChart3,
  ChevronRight, ChevronDown, RefreshCw, Sparkles, Clock
} from "lucide-react";

interface HuntResult {
  id: string;
  timestamp: string;
  source_ip: string;
  event_type: string;
  description: string;
  risk_score: number;
  matches: Record<string, unknown>;
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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
    <div className="min-h-screen bg-gray-950 text-white flex">
      {/* Sidebar */}
      <div className={`${sidebarCollapsed ? 'w-16' : 'w-80'} bg-gray-900 border-r border-gray-800 transition-all duration-300 flex flex-col`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center justify-between">
            {!sidebarCollapsed && (
              <div>
                <h1 className="text-xl font-bold text-white">SOC Command</h1>
                <p className="text-xs text-gray-400">Enterprise Security Center</p>
              </div>
            )}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {!sidebarCollapsed && (
          <>
            {/* Navigation */}
            <div className="p-4">
              <nav className="space-y-2">
                {[
                  { id: 'overview', label: 'Threat Overview', icon: BarChart3, href: '/' },
                  { id: 'incidents', label: 'Active Incidents', icon: AlertTriangle, href: '/incidents' },
                  { id: 'intelligence', label: 'Threat Intel', icon: Globe, href: '/intelligence' },
                  { id: 'hunting', label: 'Threat Hunting', icon: Target, href: '/hunt', active: true },
                  { id: 'forensics', label: 'Forensics', icon: Search, href: '/investigations' },
                  { id: 'response', label: 'Response Actions', icon: Shield, href: '/' },
                  { id: 'workflows', label: 'Workflow Automation', icon: Workflow, href: '/workflows' },
                  { id: 'visualizations', label: '3D Visualization', icon: Activity, href: '/visualizations' }
                ].map(({ id, label, icon: Icon, href, active }) => (
                  <Link
                    key={id}
                    href={href}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                      active ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30' : 'hover:bg-gray-700/50 text-gray-300 hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="text-sm font-medium">{label}</span>
                  </Link>
                ))}
              </nav>
            </div>

            {/* System Status */}
            <div className="p-4 border-t border-gray-800">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Active Threats</span>
                  <span className="text-sm font-bold text-red-400">2</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Contained</span>
                  <span className="text-sm font-bold text-green-400">1</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">AI Detected</span>
                  <span className="text-sm font-bold text-blue-400">0</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Avg Response</span>
                  <span className="text-sm font-bold text-purple-400">4.2m</span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Header Section */}
          <Card className="bg-gray-900 border-gray-800">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                    <Target className="h-8 w-8 text-green-400" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                      Threat Hunting Platform
                      <Badge className="bg-green-500/10 text-green-400 border-green-500/30">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
                        Hunt Engine Ready
                      </Badge>
                    </h1>
                    <p className="text-gray-400 text-sm mt-1 flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-green-400" />
                      Proactive threat detection and investigation
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-800 p-1">
              <TabsTrigger 
                value="hunt"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <Target className="h-4 w-4" />
                Interactive Hunt
              </TabsTrigger>
              <TabsTrigger 
                value="queries"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <Search className="h-4 w-4" />
                Saved Queries
              </TabsTrigger>
              <TabsTrigger 
                value="iocs"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <AlertTriangle className="h-4 w-4" />
                IOC Management
              </TabsTrigger>
              <TabsTrigger 
                value="analytics"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <BarChart3 className="h-4 w-4" />
                Hunt Analytics
              </TabsTrigger>
            </TabsList>

        <TabsContent value="hunt" className="space-y-6 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Query Builder */}
            <div className="lg:col-span-2">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Hunt Query Builder</CardTitle>
                  <CardDescription className="text-gray-400">
                    Execute custom threat hunting queries across all security events
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-300">Query</Label>
                    <textarea
                      value={huntQuery}
                      onChange={(e) => setHuntQuery(e.target.value)}
                      placeholder="Enter hunt query (e.g., eventid:cowrie.login.failed AND src_ip:203.0.113.*)"
                      className="w-full h-32 p-3 bg-gray-800 border border-gray-700 rounded-lg text-sm font-mono text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500/50"
                    />
                  </div>
                  
                  <div className="flex gap-2">
                    <Button 
                      onClick={runHunt} 
                      disabled={loading || !huntQuery.trim()}
                      className="bg-green-600 hover:bg-green-700 text-white"
                    >
                      {loading ? "Hunting..." : "🔍 Run Hunt"}
                    </Button>
                    
                    <Button 
                      variant="outline" 
                      onClick={() => setHuntQuery("")}
                      className="border-gray-700 text-gray-300 hover:bg-gray-800"
                    >
                      Clear
                    </Button>
                  </div>

                  {/* Quick Queries */}
                  <div className="border-t border-gray-700 pt-4">
                    <Label className="text-sm font-medium text-gray-300">Quick Hunt Templates:</Label>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("eventid:cowrie.login.failed | count by src_ip | where count > 5")}
                        className="border-gray-700 text-gray-300 hover:bg-gray-800"
                      >
                        SSH Brute Force
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("src_ip:192.168.* AND dst_ip:10.0.0.*")}
                        className="border-gray-700 text-gray-300 hover:bg-gray-800"
                      >
                        Lateral Movement
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("user_agent:/(curl|wget|python)/")}
                        className="border-gray-700 text-gray-300 hover:bg-gray-800"
                      >
                        Suspicious Agents
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setHuntQuery("timestamp:[22:00 TO 06:00]")}
                        className="border-gray-700 text-gray-300 hover:bg-gray-800"
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
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Save Query</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label className="text-gray-300">Query Name</Label>
                    <Input
                      value={queryName}
                      onChange={(e) => setQueryName(e.target.value)}
                      placeholder="e.g., Advanced SSH Hunt"
                      className="bg-gray-800 border-gray-700 text-white placeholder-gray-500"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-gray-300">Description</Label>
                    <textarea
                      value={queryDescription}
                      onChange={(e) => setQueryDescription(e.target.value)}
                      placeholder="Describe what this hunt detects..."
                      className="w-full h-20 p-2 bg-gray-800 border border-gray-700 rounded text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500/50"
                    />
                  </div>
                  
                  <Button 
                    onClick={saveQuery} 
                    disabled={!queryName.trim() || !huntQuery.trim()}
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                  >
                    💾 Save Query
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Hunt Results */}
          {huntResults.length > 0 && (
            <Card className="bg-gray-900 border-gray-800">
              <CardHeader>
                <CardTitle className="text-white">Hunt Results ({huntResults.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {huntResults.map((result) => (
                    <div key={result.id} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-sm bg-gray-800 border border-gray-700 px-2 py-1 rounded text-gray-300">
                            {result.source_ip}
                          </span>
                          <Badge variant="outline" className="border-gray-600 text-gray-300">{result.event_type}</Badge>
                          <span className={`font-semibold ${getRiskScoreColor(result.risk_score)}`}>
                            Risk: {(result.risk_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {new Date(result.timestamp).toLocaleString()}
                        </span>
                      </div>
                      
                      <p className="text-gray-300 mb-2">{result.description}</p>
                      
                      <div className="text-xs text-gray-400 bg-gray-900/50 border border-gray-700 p-2 rounded">
                        <strong className="text-gray-300">Matches:</strong> {JSON.stringify(result.matches)}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="queries" className="space-y-6 mt-6">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Saved Hunt Queries</CardTitle>
              <CardDescription className="text-gray-400">Your library of reusable threat hunting queries</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {savedQueries.length > 0 ? (
                  savedQueries.map((query, index) => (
                    <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 hover:border-gray-600 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-white">{query.name}</h3>
                        <div className="flex items-center gap-2">
                          <Badge className="bg-gray-800 border-gray-600 text-gray-300">
                            {query.risk_level.toUpperCase()}
                          </Badge>
                          <Button 
                            size="sm" 
                            onClick={() => setHuntQuery(query.query)}
                            className="bg-green-600 hover:bg-green-700 text-white"
                          >
                            Load
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-gray-400 text-sm mb-2">{query.description}</p>
                      
                      <div className="bg-gray-900/50 border border-gray-700 p-2 rounded text-xs font-mono mb-2 text-gray-300">
                        {query.query}
                      </div>
                      
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>Last run: {query.last_run ? new Date(query.last_run).toLocaleString() : "Never"}</span>
                        <span>Results: {query.results_count || 0}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No saved queries yet. Save your hunt queries for quick reuse.
                  </div>
                )}
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
