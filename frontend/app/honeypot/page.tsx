"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Shield, Activity, Database, AlertTriangle, CheckCircle, XCircle, Play, Square, Loader2, RefreshCw } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { DashboardLayout } from "../../components/DashboardLayout";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { API_BASE_URL } from "../utils/api";

const API_BASE = API_BASE_URL;

interface TPotStatus {
  status: string;
  host: string;
  monitoring_honeypots: string[];
  active_tunnels: string[];
  containers: Container[];
  blocked_ips: string[];
  blocked_count: number;
}

interface Container {
  name: string;
  status: string;
  ports: string;
}

interface RecentAttack {
  "@timestamp": string;
  src_ip: string;
  dest_ip: string;
  dest_port: number;
  honeypot?: string;
  event_type?: string;
  alert?: {
    signature: string;
    category: string;
    severity: number;
  };
}

export default function HoneypotPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [tpotStatus, setTPotStatus] = useState<TPotStatus | null>(null);
  const [recentAttacks, setRecentAttacks] = useState<RecentAttack[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/login");
    }
  }, [user, authLoading, router]);

  const fetchTPotStatus = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/tpot/status`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setTPotStatus(data);
        setError(null);
      } else {
        throw new Error("Failed to fetch T-Pot status");
      }
    } catch (err: any) {
      console.error("Error fetching T-Pot status:", err);
      setError(err.message);
    }
  };

  const fetchRecentAttacks = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/tpot/attacks/recent?minutes=5`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setRecentAttacks(data.attacks || []);
      } else if (response.status === 503) {
        // T-Pot not connected - this is expected, don't show error
        setRecentAttacks([]);
      }
    } catch (err) {
      console.error("Error fetching recent attacks:", err);
      setRecentAttacks([]);
    }
  };

  const refreshData = async () => {
    setRefreshing(true);
    await Promise.all([fetchTPotStatus(), fetchRecentAttacks()]);
    setRefreshing(false);
  };

  const reconnectTPot = async () => {
    try {
      setRefreshing(true);
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/tpot/reconnect`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          alert("Successfully reconnected to T-Pot!");
          await refreshData();
        } else {
          alert(`Reconnection failed: ${result.message || "Unknown error"}`);
        }
      } else {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        alert(`Reconnection failed: ${errorData.detail || errorData.message || "Server error"}`);
      }
    } catch (err: any) {
      console.error("Error reconnecting to T-Pot:", err);
      alert(`Failed to reconnect to T-Pot: ${err.message || "Unknown error"}`);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchTPotStatus(), fetchRecentAttacks()]);
      setLoading(false);
    };

    loadData();

    // Auto-refresh every 10 seconds
    const interval = setInterval(() => {
      fetchTPotStatus();
      fetchRecentAttacks();
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const blockIP = async (ip: string) => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/tpot/firewall/block`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ ip_address: ip }),
      });

      if (response.ok) {
        alert(`IP ${ip} blocked successfully`);
        refreshData();
      } else {
        const error = await response.json();
        alert(`Failed to block IP: ${error.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Error blocking IP:", err);
      alert("Failed to block IP");
    }
  };

  const toggleContainer = async (containerName: string, action: "start" | "stop") => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/tpot/containers/${action}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ container_name: containerName }),
      });

      if (response.ok) {
        alert(`Container ${containerName} ${action}ped successfully`);
        refreshData();
      } else {
        const error = await response.json();
        alert(`Failed to ${action} container: ${error.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error(`Error ${action}ping container:`, err);
      alert(`Failed to ${action} container`);
    }
  };

  if (authLoading || loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center min-h-screen">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </DashboardLayout>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Shield className="h-8 w-8 text-primary" />
              T-Pot Honeypot Monitoring
            </h1>
            <p className="text-muted-foreground mt-1">
              Real-time monitoring of T-Pot honeypot infrastructure
            </p>
          </div>
          <Button onClick={refreshData} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>

        {/* Error Message */}
        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-destructive">
                <XCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Connection Status</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {tpotStatus?.status === "connected" ? (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-2xl font-bold text-green-500">Connected</span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-5 w-5 text-red-500" />
                    <span className="text-2xl font-bold text-red-500">Disconnected</span>
                  </>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {tpotStatus?.host || "No host"}
              </p>
              {tpotStatus?.status !== "connected" && (
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 w-full"
                  onClick={reconnectTPot}
                  disabled={refreshing}
                >
                  {refreshing ? (
                    <>
                      <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                      Reconnecting...
                    </>
                  ) : (
                    "Reconnect"
                  )}
                </Button>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Honeypots</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {tpotStatus?.monitoring_honeypots.length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Monitoring {tpotStatus?.containers.length || 0} containers
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Blocked IPs</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{tpotStatus?.blocked_count || 0}</div>
              <p className="text-xs text-muted-foreground">Currently blocked</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Recent Attacks</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{recentAttacks.length}</div>
              <p className="text-xs text-muted-foreground">Last 5 minutes</p>
            </CardContent>
          </Card>
        </div>

        {/* Honeypot Containers */}
        <Card>
          <CardHeader>
            <CardTitle>Honeypot Containers</CardTitle>
            <CardDescription>
              Status and control of T-Pot honeypot containers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {tpotStatus?.containers.map((container) => (
                <div
                  key={container.name}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div className="flex items-center gap-4 flex-1">
                    <div className="flex items-center gap-2">
                      {container.status.includes("Up") ? (
                        <div className="h-3 w-3 rounded-full bg-green-500" />
                      ) : (
                        <div className="h-3 w-3 rounded-full bg-gray-500" />
                      )}
                      <span className="font-medium">{container.name}</span>
                    </div>
                    <Badge variant={container.status.includes("Up") ? "default" : "secondary"}>
                      {container.status}
                    </Badge>
                    {tpotStatus.monitoring_honeypots.includes(container.name) && (
                      <Badge variant="outline" className="text-green-600 border-green-600">
                        <Activity className="h-3 w-3 mr-1" />
                        Monitoring
                      </Badge>
                    )}
                  </div>
                  <div className="flex gap-2">
                    {container.status.includes("Up") ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => toggleContainer(container.name, "stop")}
                      >
                        <Square className="h-4 w-4 mr-1" />
                        Stop
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => toggleContainer(container.name, "start")}
                      >
                        <Play className="h-4 w-4 mr-1" />
                        Start
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Attacks */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Attacks (Last 5 Minutes)</CardTitle>
            <CardDescription>
              Live attack data from T-Pot honeypot sensors
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {recentAttacks.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No recent attacks detected
                </div>
              ) : (
                recentAttacks.slice(0, 10).map((attack, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-medium">{attack.src_ip}</span>
                        <span className="text-muted-foreground">→</span>
                        <span className="font-mono text-sm">
                          {attack.dest_ip}:{attack.dest_port}
                        </span>
                        {attack.honeypot && (
                          <Badge variant="outline">{attack.honeypot}</Badge>
                        )}
                      </div>
                      {attack.alert && (
                        <div className="text-sm text-muted-foreground mt-1">
                          {attack.alert.signature}
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground mt-1">
                        {new Date(attack["@timestamp"]).toLocaleString()}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => blockIP(attack.src_ip)}
                    >
                      Block IP
                    </Button>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Blocked IPs */}
        {tpotStatus && tpotStatus.blocked_ips.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Blocked IP Addresses</CardTitle>
              <CardDescription>
                Currently blocked malicious IPs on T-Pot firewall
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {tpotStatus.blocked_ips.map((ip) => (
                  <Badge key={ip} variant="destructive" className="font-mono">
                    {ip}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}
