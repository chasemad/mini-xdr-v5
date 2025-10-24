"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { DashboardLayout } from "@/components/DashboardLayout";

interface IOC {
  id: string;
  type: "ip" | "domain" | "hash" | "url" | "email" | "user_agent";
  value: string;
  confidence: number;
  threat_level: "low" | "medium" | "high" | "critical";
  source: string;
  first_seen: string;
  last_seen: string;
  tags: string[];
  description?: string;
  threat_actor?: string;
  campaign?: string;
  ttps: string[];
  false_positive: boolean;
}

interface ThreatFeed {
  id: string;
  name: string;
  url: string;
  status: "active" | "inactive" | "error";
  last_update: string;
  ioc_count: number;
  feed_type: "commercial" | "open_source" | "internal";
  enabled: boolean;
}

interface ThreatActor {
  id: string;
  name: string;
  aliases: string[];
  motivation: string;
  sophistication: "low" | "medium" | "high" | "advanced";
  regions: string[];
  sectors: string[];
  ttps: string[];
  associated_campaigns: string[];
  first_observed: string;
  last_activity: string;
}

interface Campaign {
  id: string;
  name: string;
  description: string;
  threat_actor?: string;
  start_date: string;
  end_date?: string;
  targets: string[];
  ttps: string[];
  ioc_count: number;
  status: "active" | "dormant" | "concluded";
}

export default function ThreatIntelligencePage() {
  const [activeTab, setActiveTab] = useState("iocs");
  const [iocs, setIocs] = useState<IOC[]>([]);
  const [threatFeeds, setThreatFeeds] = useState<ThreatFeed[]>([]);
  const [threatActors, setThreatActors] = useState<ThreatActor[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [newIOC, setNewIOC] = useState<{
    type: "ip" | "domain" | "hash" | "url" | "email" | "user_agent";
    value: string;
    confidence: number;
    threat_level: "low" | "medium" | "high" | "critical";
    source: string;
    description: string;
    tags: string;
  }>({
    type: "ip",
    value: "",
    confidence: 80,
    threat_level: "medium",
    source: "Manual Entry",
    description: "",
    tags: ""
  });

  // Load threat intelligence data from backend
  useEffect(() => {
    const loadIOCs = async () => {
      try {
        const response = await fetch('/api/intelligence/iocs');
        if (response.ok) {
          const data = await response.json();
          setIocs(data.iocs || []);
        }
      } catch (error) {
        console.error('Failed to load IOCs:', error);
      }
    };

    const loadThreatFeeds = async () => {
      try {
        const response = await fetch('/api/intelligence/feeds');
        if (response.ok) {
          const data = await response.json();
          setThreatFeeds(data.feeds || []);
        }
      } catch (error) {
        console.error('Failed to load threat feeds:', error);
      }
    };

    const loadThreatActors = async () => {
      try {
        const response = await fetch('/api/intelligence/actors');
        if (response.ok) {
          const data = await response.json();
          setThreatActors(data.actors || []);
        }
      } catch (error) {
        console.error('Failed to load threat actors:', error);
      }
    };

    const loadCampaigns = async () => {
      try {
        const response = await fetch('/api/intelligence/campaigns');
        if (response.ok) {
          const data = await response.json();
          setCampaigns(data.campaigns || []);
        }
      } catch (error) {
        console.error('Failed to load campaigns:', error);
      }
    };

    loadIOCs();
    loadThreatFeeds();
    loadThreatActors();
    loadCampaigns();
  }, []);

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
          confidence: newIOC.confidence / 100,
          threat_level: newIOC.threat_level,
          source: newIOC.source,
          description: newIOC.description,
          tags: newIOC.tags.split(",").map(t => t.trim()).filter(t => t)
        }),
      });

      if (response.ok) {
        const ioc = await response.json();
        setIocs(prev => [...prev, ioc]);
        setNewIOC({
          type: "ip",
          value: "",
          confidence: 80,
          threat_level: "medium",
          source: "Manual Entry",
          description: "",
          tags: ""
        });
      }
    } catch (error) {
      console.error('Failed to add IOC:', error);
    }
  };

  const filteredIOCs = iocs.filter(ioc => {
    const matchesSearch = ioc.value.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         ioc.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         ioc.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesType = filterType === "all" || ioc.type === filterType;
    return matchesSearch && matchesType;
  });

  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case "critical": return "bg-red-100 text-red-800 border-red-200";
      case "high": return "bg-orange-100 text-orange-800 border-orange-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "bg-green-100 text-green-800 border-green-200";
      case "inactive": return "bg-gray-100 text-gray-800 border-gray-200";
      case "error": return "bg-red-100 text-red-800 border-red-200";
      case "dormant": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "concluded": return "bg-blue-100 text-blue-800 border-blue-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const formatTTPs = (ttps: string[]) => {
    return ttps.map(ttp => (
      <Badge key={ttp} variant="outline" className="text-xs">
        {ttp}
      </Badge>
    ));
  };

  return (
    <DashboardLayout breadcrumbs={[{ label: "Threat Intelligence" }]}>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Threat Intelligence</h1>
            <p className="text-gray-400 mt-1">Centralized threat intelligence management and analysis</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-300">Intel Feeds Active</span>
          </div>
        </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="iocs">IOCs</TabsTrigger>
          <TabsTrigger value="feeds">Threat Feeds</TabsTrigger>
          <TabsTrigger value="actors">Threat Actors</TabsTrigger>
          <TabsTrigger value="campaigns">Campaigns</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="iocs" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* IOC Management */}
            <div className="lg:col-span-3">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Indicators of Compromise ({filteredIOCs.length})</CardTitle>
                    <div className="flex gap-2">
                      <Input
                        placeholder="Search IOCs..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-48"
                      />
                      <Select value={filterType} onValueChange={setFilterType}>
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Types</SelectItem>
                          <SelectItem value="ip">IP</SelectItem>
                          <SelectItem value="domain">Domain</SelectItem>
                          <SelectItem value="hash">Hash</SelectItem>
                          <SelectItem value="url">URL</SelectItem>
                          <SelectItem value="email">Email</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {filteredIOCs.map((ioc) => (
                      <div key={ioc.id} className={`border rounded-lg p-4 ${ioc.false_positive ? 'bg-gray-50 opacity-75' : ''}`}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <Badge variant="outline">{ioc.type.toUpperCase()}</Badge>
                            <span className="font-mono text-sm font-medium">{ioc.value}</span>
                            <Badge className={`${getThreatLevelColor(ioc.threat_level)} border`}>
                              {ioc.threat_level.toUpperCase()}
                            </Badge>
                            {ioc.false_positive && (
                              <Badge className="bg-gray-100 text-gray-800 border-gray-200">
                                FALSE POSITIVE
                              </Badge>
                            )}
                          </div>
                          <div className="text-right">
                            <span className={`text-sm font-medium ${getConfidenceColor(ioc.confidence)}`}>
                              {(ioc.confidence * 100).toFixed(0)}% confidence
                            </span>
                          </div>
                        </div>

                        {ioc.description && (
                          <p className="text-gray-700 text-sm mb-2">{ioc.description}</p>
                        )}

                        <div className="grid grid-cols-2 gap-4 text-xs text-gray-600 mb-2">
                          <div>
                            <span className="font-medium">Source:</span> {ioc.source}
                          </div>
                          <div>
                            <span className="font-medium">First Seen:</span> {new Date(ioc.first_seen).toLocaleDateString()}
                          </div>
                          {ioc.threat_actor && (
                            <div>
                              <span className="font-medium">Threat Actor:</span> {ioc.threat_actor}
                            </div>
                          )}
                          {ioc.campaign && (
                            <div>
                              <span className="font-medium">Campaign:</span> {ioc.campaign}
                            </div>
                          )}
                        </div>

                        {ioc.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mb-2">
                            {ioc.tags.map((tag, index) => (
                              <Badge key={index} variant="outline" className="text-xs">
                                #{tag}
                              </Badge>
                            ))}
                          </div>
                        )}

                        {ioc.ttps.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            <span className="text-xs text-gray-600 mr-2">TTPs:</span>
                            {formatTTPs(ioc.ttps)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Add IOC Panel */}
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
                      onValueChange={(value: "ip" | "domain" | "hash" | "url" | "email" | "user_agent") => setNewIOC({...newIOC, type: value})}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ip">IP Address</SelectItem>
                        <SelectItem value="domain">Domain</SelectItem>
                        <SelectItem value="hash">File Hash</SelectItem>
                        <SelectItem value="url">URL</SelectItem>
                        <SelectItem value="email">Email</SelectItem>
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
                    <Label>Threat Level</Label>
                    <Select
                      value={newIOC.threat_level}
                      onValueChange={(value: "low" | "medium" | "high" | "critical") => setNewIOC({...newIOC, threat_level: value})}
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
                    <Label>Confidence ({newIOC.confidence}%)</Label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={newIOC.confidence}
                      onChange={(e) => setNewIOC({...newIOC, confidence: parseInt(e.target.value)})}
                      className="w-full"
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

                  <div className="space-y-2">
                    <Label>Description</Label>
                    <textarea
                      value={newIOC.description}
                      onChange={(e) => setNewIOC({...newIOC, description: e.target.value})}
                      placeholder="Description of the threat..."
                      className="w-full h-20 p-2 border border-gray-300 rounded text-sm"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Tags (comma-separated)</Label>
                    <Input
                      value={newIOC.tags}
                      onChange={(e) => setNewIOC({...newIOC, tags: e.target.value})}
                      placeholder="malware, c2, botnet"
                    />
                  </div>

                  <Button onClick={addIOC} disabled={!newIOC.value.trim()} className="w-full">
                    Add IOC
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="feeds" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Threat Intelligence Feeds</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {threatFeeds.map((feed) => (
                  <div key={feed.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <h3 className="font-semibold">{feed.name}</h3>
                        <Badge className={`${getStatusColor(feed.status)} border`}>
                          {feed.status.toUpperCase()}
                        </Badge>
                        <Badge variant="outline">{feed.feed_type.replace("_", " ").toUpperCase()}</Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-600">
                          {feed.ioc_count.toLocaleString()} IOCs
                        </span>
                        <Button size="sm" variant={feed.enabled ? "outline" : "default"}>
                          {feed.enabled ? "Disable" : "Enable"}
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                      <div>
                        <span className="font-medium">URL:</span> {feed.url}
                      </div>
                      <div>
                        <span className="font-medium">Last Update:</span> {new Date(feed.last_update).toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="actors" className="space-y-6">
          <div className="grid gap-4">
            {threatActors.map((actor) => (
              <Card key={actor.id}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-lg">{actor.name}</h3>
                      <Badge className={`${getStatusColor(actor.sophistication)} border`}>
                        {actor.sophistication.toUpperCase()} SOPHISTICATION
                      </Badge>
                    </div>
                    <span className="text-sm text-gray-600">ID: {actor.id}</span>
                  </div>

                  {actor.aliases.length > 0 && (
                    <div className="mb-2">
                      <span className="text-sm font-medium text-gray-700">Aliases: </span>
                      <span className="text-sm text-gray-600">{actor.aliases.join(", ")}</span>
                    </div>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                    <div>
                      <span className="font-medium text-gray-700">Motivation:</span>
                      <p className="text-gray-600">{actor.motivation}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Regions:</span>
                      <p className="text-gray-600">{actor.regions.join(", ")}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">First Observed:</span>
                      <p className="text-gray-600">{new Date(actor.first_observed).getFullYear()}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Last Activity:</span>
                      <p className="text-gray-600">{new Date(actor.last_activity).toLocaleDateString()}</p>
                    </div>
                  </div>

                  <div className="mb-3">
                    <span className="text-sm font-medium text-gray-700">Target Sectors: </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {actor.sectors.map((sector, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {sector}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="mb-3">
                    <span className="text-sm font-medium text-gray-700">TTPs: </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {formatTTPs(actor.ttps)}
                    </div>
                  </div>

                  {actor.associated_campaigns.length > 0 && (
                    <div>
                      <span className="text-sm font-medium text-gray-700">Associated Campaigns: </span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {actor.associated_campaigns.map((campaign, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {campaign}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="campaigns" className="space-y-6">
          <div className="grid gap-4">
            {campaigns.map((campaign) => (
              <Card key={campaign.id}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-lg">{campaign.name}</h3>
                      <Badge className={`${getStatusColor(campaign.status)} border`}>
                        {campaign.status.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">{campaign.id}</p>
                      <p className="text-sm text-gray-600">{campaign.ioc_count} IOCs</p>
                    </div>
                  </div>

                  <p className="text-gray-700 mb-3">{campaign.description}</p>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                    <div>
                      <span className="font-medium text-gray-700">Start Date:</span>
                      <p className="text-gray-600">{new Date(campaign.start_date).toLocaleDateString()}</p>
                    </div>
                    {campaign.end_date && (
                      <div>
                        <span className="font-medium text-gray-700">End Date:</span>
                        <p className="text-gray-600">{new Date(campaign.end_date).toLocaleDateString()}</p>
                      </div>
                    )}
                    {campaign.threat_actor && (
                      <div>
                        <span className="font-medium text-gray-700">Threat Actor:</span>
                        <p className="text-gray-600">{campaign.threat_actor}</p>
                      </div>
                    )}
                    <div>
                      <span className="font-medium text-gray-700">Duration:</span>
                      <p className="text-gray-600">
                        {campaign.end_date ?
                          `${Math.ceil((new Date(campaign.end_date).getTime() - new Date(campaign.start_date).getTime()) / (1000 * 60 * 60 * 24))} days` :
                          `${Math.ceil((new Date().getTime() - new Date(campaign.start_date).getTime()) / (1000 * 60 * 60 * 24))} days (ongoing)`
                        }
                      </p>
                    </div>
                  </div>

                  <div className="mb-3">
                    <span className="text-sm font-medium text-gray-700">Targets: </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {campaign.targets.map((target, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {target}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div>
                    <span className="text-sm font-medium text-gray-700">TTPs: </span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {formatTTPs(campaign.ttps)}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Intelligence Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total IOCs</span>
                    <span className="font-semibold">{iocs.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">High/Critical</span>
                    <span className="font-semibold text-red-600">
                      {iocs.filter(i => ["high", "critical"].includes(i.threat_level)).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Active Feeds</span>
                    <span className="font-semibold">{threatFeeds.filter(f => f.status === "active").length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">False Positives</span>
                    <span className="font-semibold">{iocs.filter(i => i.false_positive).length}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>IOC Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {["ip", "domain", "hash", "url"].map(type => {
                    const count = iocs.filter(i => i.type === type).length;
                    const percentage = (count / iocs.length) * 100;
                    return (
                      <div key={type} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{type}</span>
                          <span>{count}</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Threat Actors</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {threatActors.map(actor => (
                    <div key={actor.id} className="flex justify-between items-center">
                      <span className="text-sm">{actor.name}</span>
                      <Badge className={`${getStatusColor(actor.sophistication)} text-xs`}>
                        {actor.sophistication}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="border-l-2 border-red-500 pl-2">
                    <p className="font-medium">New critical IOC</p>
                    <p className="text-gray-500 text-xs">15 minutes ago</p>
                  </div>
                  <div className="border-l-2 border-blue-500 pl-2">
                    <p className="font-medium">Feed updated</p>
                    <p className="text-gray-500 text-xs">1 hour ago</p>
                  </div>
                  <div className="border-l-2 border-green-500 pl-2">
                    <p className="font-medium">IOC marked FP</p>
                    <p className="text-gray-500 text-xs">2 hours ago</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
      </div>
    </DashboardLayout>
  );
}
