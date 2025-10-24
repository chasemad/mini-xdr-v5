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
  ChevronRight, ChevronDown, RefreshCw, Sparkles, Database, Clock, TrendingUp
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

export default function ThreatHuntingPage() {
  const [activeTab, setActiveTab] = useState("hunt");
  const [huntQuery, setHuntQuery] = useState("");
  const [huntResults, setHuntResults] = useState<HuntResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [queryName, setQueryName] = useState("");
  const [queryDescription, setQueryDescription] = useState("");

  const runHunt = async () => {
    if (!huntQuery.trim()) return;
    
    setLoading(true);
    try {
      // Simulate hunt execution (replace with real API call)
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock results (replace with real API response)
      setHuntResults([
        {
          id: '1',
          timestamp: new Date().toISOString(),
          source_ip: '192.168.100.99',
          event_type: 'cowrie.login.failed',
          description: 'Multiple failed SSH login attempts detected',
          risk_score: 0.85,
          matches: { attempts: 45, usernames: ['root', 'admin', 'user'] }
        }
      ]);
    } catch (error) {
      console.error("Hunt execution failed:", error);
      setHuntResults([]);
    } finally {
      setLoading(false);
    }
  };

  const getRiskScoreColor = (score: number) => {
    if (score >= 0.8) return "text-red-400";
    if (score >= 0.6) return "text-orange-400";
    if (score >= 0.4) return "text-yellow-400";
    return "text-green-400";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-black text-white flex">
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
                  <span className="text-xs text-gray-400">Hunt Queries</span>
                  <span className="text-sm font-bold text-blue-400">15</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">IOCs Tracked</span>
                  <span className="text-sm font-bold text-purple-400">42</span>
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

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-green-400">47</div>
                    <div className="text-xs text-gray-400 mt-1">Total Hunts</div>
                  </div>
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-blue-400">12</div>
                    <div className="text-xs text-gray-400 mt-1">This Week</div>
                  </div>
                  <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-purple-400">85%</div>
                    <div className="text-xs text-gray-400 mt-1">Success Rate</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Main Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 bg-gray-900 border border-gray-800 p-1">
              <TabsTrigger 
                value="hunt"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <Target className="h-4 w-4" />
                <span className="hidden md:inline">Interactive Hunt</span>
              </TabsTrigger>
              <TabsTrigger 
                value="queries"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <Database className="h-4 w-4" />
                <span className="hidden md:inline">Saved Queries</span>
              </TabsTrigger>
              <TabsTrigger 
                value="iocs"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <AlertTriangle className="h-4 w-4" />
                <span className="hidden md:inline">IOCs</span>
              </TabsTrigger>
              <TabsTrigger 
                value="analytics"
                className="flex items-center gap-2 data-[state=active]:bg-green-600 data-[state=active]:text-white text-gray-400"
              >
                <TrendingUp className="h-4 w-4" />
                <span className="hidden md:inline">Analytics</span>
              </TabsTrigger>
            </TabsList>

            {/* Interactive Hunt Tab */}
            <TabsContent value="hunt" className="space-y-6 mt-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                          {loading ? (
                            <><RefreshCw className="w-4 h-4 animate-spin mr-2" />Hunting...</>
                          ) : (
                            <>🔍 Run Hunt</>
                          )}
                        </Button>
                        
                        <Button 
                          variant="outline" 
                          onClick={() => setHuntQuery("")}
                          className="border-gray-700 text-gray-300 hover:bg-gray-800"
                        >
                          Clear
                        </Button>
                      </div>

                      {/* Quick Templates */}
                      <div className="border-t border-gray-700 pt-4">
                        <Label className="text-sm font-medium text-gray-300 mb-2">Quick Hunt Templates:</Label>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          {[
                            { label: 'SSH Brute Force', query: 'eventid:cowrie.login.failed | count by src_ip | where count > 5' },
                            { label: 'Lateral Movement', query: 'src_ip:192.168.* AND dst_ip:10.0.0.*' },
                            { label: 'Suspicious Agents', query: 'user_agent:/(curl|wget|python)/' },
                            { label: 'Off-Hours Activity', query: 'timestamp:[22:00 TO 06:00]' }
                          ].map((template) => (
                            <Button
                              key={template.label}
                              variant="outline"
                              size="sm"
                              onClick={() => setHuntQuery(template.query)}
                              className="border-gray-700 text-gray-300 hover:bg-gray-800"
                            >
                              {template.label}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Save Query Panel */}
                <div>
                  <Card className="bg-gray-900 border-gray-800">
                    <CardHeader>
                      <CardTitle className="text-white">Save Hunt Query</CardTitle>
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
                        onClick={() => alert('Query saved!')} 
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
                              <span className="font-mono text-sm bg-gray-900/50 border border-gray-700 px-2 py-1 rounded text-gray-300">
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

            {/* Saved Queries Tab */}
            <TabsContent value="queries" className="space-y-6 mt-6">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Saved Hunt Queries</CardTitle>
                  <CardDescription className="text-gray-400">Your library of reusable threat hunting queries</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-12 text-gray-500">
                    <Database className="w-12 h-12 mx-auto mb-3 text-gray-600" />
                    <p>No saved queries yet</p>
                    <p className="text-sm mt-1">Save your hunt queries for quick reuse</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* IOCs Tab */}
            <TabsContent value="iocs" className="space-y-6 mt-6">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Indicators of Compromise (IOCs)</CardTitle>
                  <CardDescription className="text-gray-400">Track and manage threat indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-12 text-gray-500">
                    <AlertTriangle className="w-12 h-12 mx-auto mb-3 text-gray-600" />
                    <p>No IOCs tracked yet</p>
                    <p className="text-sm mt-1">IOCs will be automatically extracted from hunt results</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics" className="space-y-6 mt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="bg-gray-900 border-green-500/30">
                  <CardContent className="p-6 text-center">
                    <TrendingUp className="w-8 h-8 text-green-400 mx-auto mb-3" />
                    <div className="text-3xl font-bold text-green-400 mb-1">47</div>
                    <div className="text-sm text-gray-400">Total Hunts</div>
                    <div className="text-xs text-gray-500 mt-1">Last 30 days</div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-900 border-blue-500/30">
                  <CardContent className="p-6 text-center">
                    <Activity className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                    <div className="text-3xl font-bold text-blue-400 mb-1">12</div>
                    <div className="text-sm text-gray-400">This Week</div>
                    <div className="text-xs text-gray-500 mt-1">↑ 15% from last week</div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-900 border-purple-500/30">
                  <CardContent className="p-6 text-center">
                    <Target className="w-8 h-8 text-purple-400 mx-auto mb-3" />
                    <div className="text-3xl font-bold text-purple-400 mb-1">85.1%</div>
                    <div className="text-sm text-gray-400">Success Rate</div>
                    <div className="text-xs text-gray-500 mt-1">Avg 8.3 results/hunt</div>
                  </CardContent>
                </Card>
              </div>

              {/* Top Threats */}
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="text-white">Top Threats Found</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {[
                      { name: 'SSH Brute Force', count: 15, color: 'red' },
                      { name: 'Port Scanning', count: 8, color: 'orange' },
                      { name: 'Lateral Movement', count: 3, color: 'yellow' },
                      { name: 'Suspicious User Agents', count: 12, color: 'blue' }
                    ].map((threat) => (
                      <div key={threat.name} className="flex items-center justify-between py-2">
                        <span className="text-gray-300">{threat.name}</span>
                        <Badge className={`bg-${threat.color}-500/20 text-${threat.color}-300 border-${threat.color}-500/30`}>
                          {threat.count}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

