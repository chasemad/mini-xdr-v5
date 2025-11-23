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
import { Slider } from "@/components/ui/slider";
import { DashboardLayout } from "@/components/DashboardLayout";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface SystemSetting {
  key: string;
  value: string | number | boolean;
  type: "string" | "number" | "boolean" | "select";
  options?: string[];
  description: string;
  category: string;
  sensitive?: boolean;
}

interface DetectionRule {
  id: string;
  name: string;
  description: string;
  type: "threshold" | "pattern" | "ml" | "correlation";
  enabled: boolean;
  severity: "low" | "medium" | "high" | "critical";
  conditions: Record<string, unknown>;
  actions: string[];
}

interface Integration {
  id: string;
  name: string;
  type: "siem" | "soar" | "threat_intel" | "notification";
  status: "connected" | "disconnected" | "error";
  config: Record<string, unknown>;
  last_sync?: string;
}

interface User {
  id: string;
  username: string;
  email: string;
  role: "admin" | "analyst" | "viewer";
  status: "active" | "inactive";
  last_login?: string;
  created_at: string;
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("system");
  const [settings, setSettings] = useState<SystemSetting[]>([]);
  const [rules, setRules] = useState<DetectionRule[]>([]);
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [newUser, setNewUser] = useState<{
    username: string;
    email: string;
    role: "admin" | "analyst" | "viewer";
  }>({
    username: "",
    email: "",
    role: "analyst"
  });

  // Load system configuration data from backend
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await fetch('/api/settings/system');
        if (response.ok) {
          const data = await response.json();
          setSettings(data.settings || []);
        }
      } catch (error) {
        console.error('Failed to load system settings:', error);
      }
    };

    const loadRules = async () => {
      try {
        const response = await fetch('/api/settings/rules');
        if (response.ok) {
          const data = await response.json();
          setRules(data.rules || []);
        }
      } catch (error) {
        console.error('Failed to load detection rules:', error);
      }
    };

    const loadIntegrations = async () => {
      try {
        const response = await fetch('/api/settings/integrations');
        if (response.ok) {
          const data = await response.json();
          setIntegrations(data.integrations || []);
        }
      } catch (error) {
        console.error('Failed to load integrations:', error);
      }
    };

    const loadUsers = async () => {
      try {
        const response = await fetch('/api/settings/users');
        if (response.ok) {
          const data = await response.json();
          setUsers(data.users || []);
        }
      } catch (error) {
        console.error('Failed to load users:', error);
      }
    };

    loadSettings();
    loadRules();
    loadIntegrations();
    loadUsers();
  }, []);

  const updateSetting = async (key: string, value: string | number | boolean) => {
    try {
      const response = await fetch('/api/settings/system', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key, value }),
      });

      if (response.ok) {
        setSettings(prev => prev.map(setting =>
          setting.key === key ? { ...setting, value } : setting
        ));
      }
    } catch (error) {
      console.error('Failed to update setting:', error);
    }
  };

  const toggleRule = async (ruleId: string) => {
    try {
      const response = await fetch(`/api/settings/rules/${ruleId}/toggle`, {
        method: 'PUT',
      });

      if (response.ok) {
        setRules(prev => prev.map(rule =>
          rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
        ));
      }
    } catch (error) {
      console.error('Failed to toggle rule:', error);
    }
  };

  const addUser = async () => {
    if (!newUser.username.trim() || !newUser.email.trim()) return;

    try {
      const response = await fetch('/api/settings/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: newUser.username,
          email: newUser.email,
          role: newUser.role
        }),
      });

      if (response.ok) {
        const user = await response.json();
        setUsers(prev => [...prev, user]);
        setNewUser({ username: "", email: "", role: "analyst" });
      }
    } catch (error) {
      console.error('Failed to add user:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "connected":
      case "active": return "bg-green-100 text-green-800 border-green-200";
      case "disconnected":
      case "inactive": return "bg-gray-100 text-gray-800 border-gray-200";
      case "error": return "bg-red-100 text-red-800 border-red-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical": return "bg-red-100 text-red-800 border-red-200";
      case "high": return "bg-orange-100 text-orange-800 border-orange-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case "admin": return "bg-purple-100 text-purple-800 border-purple-200";
      case "analyst": return "bg-blue-100 text-blue-800 border-blue-200";
      case "viewer": return "bg-gray-100 text-gray-800 border-gray-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const renderSettingInput = (setting: SystemSetting) => {
    switch (setting.type) {
      case "boolean":
        return (
          <button
            onClick={() => updateSetting(setting.key, !setting.value)}
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-offset-2 ${
              setting.value ? "bg-indigo-600" : "bg-gray-200"
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                setting.value ? "translate-x-5" : "translate-x-0"
              }`}
            />
          </button>
        );
      case "select":
        return (
          <Select
            value={String(setting.value)}
            onValueChange={(value) => updateSetting(setting.key, value)}
          >
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {setting.options?.map(option => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      case "number":
        if (setting.key === "ml_confidence_threshold") {
          return (
            <div className="w-48">
              <Slider
                value={[Number(setting.value)]}
                onValueChange={([value]) => updateSetting(setting.key, value)}
                max={1}
                min={0}
                step={0.01}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">
                Value: {Number(setting.value).toFixed(2)}
              </div>
            </div>
          );
        }
        return (
          <Input
            type="number"
            value={String(setting.value)}
            onChange={(e) => updateSetting(setting.key, parseInt(e.target.value) || 0)}
            className="w-32"
          />
        );
      default:
        return (
          <Input
            type={setting.sensitive ? "password" : "text"}
            value={String(setting.value)}
            onChange={(e) => updateSetting(setting.key, e.target.value)}
            className="w-64"
          />
        );
    }
  };

  return (
    <DashboardLayout breadcrumbs={[{ label: "Settings" }]}>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">System Settings</h1>
            <p className="text-gray-400 mt-1">Configure system behavior, rules, and integrations</p>
          </div>
          <Button>
            Save All Changes
          </Button>
        </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="system">System</TabsTrigger>
          <TabsTrigger value="rules">Detection Rules</TabsTrigger>
          <TabsTrigger value="integrations">Integrations</TabsTrigger>
          <TabsTrigger value="users">Users & Access</TabsTrigger>
          <TabsTrigger value="backup">Backup & Recovery</TabsTrigger>
        </TabsList>

        <TabsContent value="system" className="space-y-6">
          {/* Group settings by category */}
          {["Detection", "ML Engine", "AI Integration", "Threat Intelligence", "Storage", "Notifications"].map(category => {
            const categorySettings = settings.filter(s => s.category === category);
            if (categorySettings.length === 0) return null;

            return (
              <Card key={category}>
                <CardHeader>
                  <CardTitle>{category} Settings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {categorySettings.map((setting) => (
                      <div key={setting.key} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                        <div className="flex-1">
                          <h3 className="font-medium capitalize">
                            {setting.key.replace(/_/g, " ")}
                          </h3>
                          <p className="text-sm text-gray-600">{setting.description}</p>
                        </div>
                        <div className="ml-4">
                          {renderSettingInput(setting)}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </TabsContent>

        <TabsContent value="rules" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Detection Rules ({rules.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Severity</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Actions</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rules.map((rule) => (
                      <TableRow key={rule.id}>
                        <TableCell className="font-medium">
                          <div>{rule.name}</div>
                          <div className="text-xs text-muted-foreground">{rule.description}</div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{rule.type.toUpperCase()}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge className={`${getSeverityColor(rule.severity)} border`}>
                            {rule.severity.toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge className={`${getStatusColor(rule.enabled ? "active" : "inactive")} border`}>
                            {rule.enabled ? "ENABLED" : "DISABLED"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {rule.actions.map((action, index) => (
                              <Badge key={index} variant="secondary" className="text-xs px-1">
                                {action.replace(/_/g, " ")}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button size="sm" variant="outline">
                              Edit
                            </Button>
                            <Button
                              size="sm"
                              variant={rule.enabled ? "destructive" : "default"}
                              onClick={() => toggleRule(rule.id)}
                            >
                              {rule.enabled ? "Disable" : "Enable"}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integrations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>External Integrations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {integrations.map((integration) => (
                  <div key={integration.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <h3 className="font-semibold">{integration.name}</h3>
                        <Badge variant="outline">{integration.type.replace("_", " ").toUpperCase()}</Badge>
                        <Badge className={`${getStatusColor(integration.status)} border`}>
                          {integration.status.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button size="sm" variant="outline">
                          Configure
                        </Button>
                        <Button size="sm">
                          Test Connection
                        </Button>
                      </div>
                    </div>

                    {integration.last_sync && (
                      <p className="text-sm text-gray-600 mb-2">
                        Last sync: {new Date(integration.last_sync).toLocaleString()}
                      </p>
                    )}

                    <div className="bg-gray-50 p-3 rounded-lg">
                      <h4 className="text-sm font-medium mb-2">Configuration:</h4>
                      <div className="text-xs text-gray-700 space-y-1">
                        {Object.entries(integration.config).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="font-medium">{key}:</span>
                            <span className="font-mono">
                              {key.toLowerCase().includes("key") || key.toLowerCase().includes("password") ?
                                "***************" : String(value)
                              }
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* User List */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>User Accounts ({users.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md border overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>User</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {users.map((user) => (
                          <TableRow key={user.id}>
                            <TableCell>
                              <div className="font-medium">{user.username}</div>
                              <div className="text-xs text-muted-foreground">{user.email}</div>
                            </TableCell>
                            <TableCell>
                              <Badge className={`${getRoleColor(user.role)} border`}>
                                {user.role.toUpperCase()}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge className={`${getStatusColor(user.status)} border`}>
                                {user.status.toUpperCase()}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-2">
                                <Button size="sm" variant="outline">
                                  Edit
                                </Button>
                                {user.role !== "admin" && (
                                  <Button size="sm" variant="destructive">
                                    Disable
                                  </Button>
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Add User Panel */}
            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Add User</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Username</Label>
                    <Input
                      value={newUser.username}
                      onChange={(e) => setNewUser({...newUser, username: e.target.value})}
                      placeholder="Enter username"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Email</Label>
                    <Input
                      type="email"
                      value={newUser.email}
                      onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                      placeholder="Enter email address"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Role</Label>
                    <Select
                      value={newUser.role}
                      onValueChange={(value: "admin" | "analyst" | "viewer") => setNewUser({...newUser, role: value})}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="viewer">Viewer</SelectItem>
                        <SelectItem value="analyst">Analyst</SelectItem>
                        <SelectItem value="admin">Admin</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="text-xs text-gray-600 space-y-1">
                    <p><strong>Viewer:</strong> Read-only access</p>
                    <p><strong>Analyst:</strong> Can investigate and respond</p>
                    <p><strong>Admin:</strong> Full system access</p>
                  </div>

                  <Button
                    onClick={addUser}
                    disabled={!newUser.username.trim() || !newUser.email.trim()}
                    className="w-full"
                  >
                    Add User
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="backup" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>System Backup</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Last Backup:</span>
                    <span className="font-medium">2024-01-15 02:00 AM</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Backup Size:</span>
                    <span className="font-medium">2.4 GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <Badge className="bg-green-100 text-green-800 border-green-200">
                      HEALTHY
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <Button className="w-full">
                    Create Backup Now
                  </Button>
                  <Button variant="outline" className="w-full">
                    View Backup History
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Data Recovery</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Alert>
                  <AlertDescription>
                    Recovery operations should only be performed by system administrators.
                  </AlertDescription>
                </Alert>

                <div className="space-y-2">
                  <Label>Select Backup</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Choose backup to restore" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="backup-20240115">2024-01-15 02:00 AM</SelectItem>
                      <SelectItem value="backup-20240114">2024-01-14 02:00 AM</SelectItem>
                      <SelectItem value="backup-20240113">2024-01-13 02:00 AM</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Button variant="destructive" className="w-full">
                    Restore System
                  </Button>
                  <Button variant="outline" className="w-full">
                    Export Configuration
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Maintenance Tasks</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h3 className="font-semibold mb-2">Database Cleanup</h3>
                  <p className="text-sm text-gray-600 mb-3">Remove old logs and optimize database</p>
                  <Button size="sm" className="w-full">
                    Run Cleanup
                  </Button>
                </div>

                <div className="p-4 border border-gray-200 rounded-lg">
                  <h3 className="font-semibold mb-2">Cache Clear</h3>
                  <p className="text-sm text-gray-600 mb-3">Clear system caches and temporary files</p>
                  <Button size="sm" variant="outline" className="w-full">
                    Clear Cache
                  </Button>
                </div>

                <div className="p-4 border border-gray-200 rounded-lg">
                  <h3 className="font-semibold mb-2">System Health Check</h3>
                  <p className="text-sm text-gray-600 mb-3">Verify all components are functioning</p>
                  <Button size="sm" variant="outline" className="w-full">
                    Run Diagnostics
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      </div>
    </DashboardLayout>
  );
}
