"use client";

import { useState, useEffect } from "react";
import {
  Shield, AlertTriangle, Bot, Zap, Play, Pause, Archive, Edit, Eye, Trash2,
  CheckCircle, XCircle, Clock, Settings, Filter, Search, ChevronDown, ChevronRight,
  FileText, TrendingUp, Activity, Target, Workflow, Plus, RefreshCw
} from "lucide-react";
import Link from "next/link";
import { DashboardLayout } from "@/components/DashboardLayout";
import { apiUrl } from "@/app/utils/api";

interface WorkflowTrigger {
  id: number;
  name: string;
  description?: string;
  category: string;
  enabled: boolean;
  auto_execute: boolean;
  priority: string;
  status: string;
  source: string;
  source_prompt?: string;
  parser_confidence?: number;
  request_type?: string;
  playbook_name: string;
  workflow_steps: any[];
  trigger_count: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
  last_triggered_at?: string;
  last_run_status?: string;
  owner?: string;
  last_editor?: string;
  created_at: string;
  updated_at: string;
}

interface NLPSuggestion {
  id: number;
  prompt: string;
  incident_id?: number;
  request_type: string;
  priority: string;
  confidence: number;
  fallback_used: boolean;
  workflow_steps: any[];
  detected_actions?: string[];
  status: string;
  created_at: string;
}

export default function AutomationsPage() {
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "demo-minixdr-api-key";

  const [activeTab, setActiveTab] = useState<"active" | "suggestions" | "archived" | "insights">("active");
  const [triggers, setTriggers] = useState<WorkflowTrigger[]>([]);
  const [suggestions, setSuggestions] = useState<NLPSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTrigger, setSelectedTrigger] = useState<WorkflowTrigger | null>(null);
  const [settingsTrigger, setSettingsTrigger] = useState<WorkflowTrigger | null>(null);
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSource, setFilterSource] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedItems, setSelectedItems] = useState<number[]>([]);

  useEffect(() => {
    fetchTriggers();
    fetchSuggestions();
  }, []);

  const fetchTriggers = async () => {
    try {
      const response = await fetch(apiUrl("/api/triggers/"), {
        headers: { "X-API-Key": API_KEY }
      });
      const data = await response.json();
      setTriggers(data);
    } catch (error) {
      console.error("Failed to fetch triggers:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSuggestions = async () => {
    try {
      const response = await fetch(apiUrl("/api/nlp-suggestions/"), {
        headers: { "X-API-Key": API_KEY }
      });
      const data = await response.json();
      setSuggestions(data);
    } catch (error) {
      console.error("Failed to fetch suggestions:", error);
    }
  };

  const handleBulkAction = async (action: "pause" | "resume" | "archive") => {
    try {
      await fetch(apiUrl(`/api/triggers/bulk/${action}`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY
        },
        body: JSON.stringify(selectedItems)
      });
      setSelectedItems([]);
      fetchTriggers();
    } catch (error) {
      console.error(`Bulk ${action} failed:`, error);
    }
  };

  const updateTriggerSettings = async (triggerId: number, settings: any) => {
    try {
      await fetch(apiUrl(`/api/triggers/${triggerId}/settings`), {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY
        },
        body: JSON.stringify(settings)
      });
      fetchTriggers();
      setSettingsTrigger(null);
    } catch (error) {
      console.error("Failed to update settings:", error);
    }
  };

  const getStatusBadge = (status: string, enabled: boolean) => {
    if (status === "archived") return <span className="px-2 py-1 rounded-full text-xs bg-gray-500/20 text-gray-400">Archived</span>;
    if (!enabled || status === "paused") return <span className="px-2 py-1 rounded-full text-xs bg-yellow-500/20 text-yellow-400">Paused</span>;
    if (status === "error") return <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">Error</span>;
    return <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">Active</span>;
  };

  const getSourceBadge = (source: string) => {
    const badges = {
      nlp: <span className="px-2 py-1 rounded-full text-xs bg-purple-500/20 text-purple-400">NLP</span>,
      manual: <span className="px-2 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400">Manual</span>,
      template: <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">Template</span>,
      api: <span className="px-2 py-1 rounded-full text-xs bg-orange-500/20 text-orange-400">API</span>
    };
    return badges[source as keyof typeof badges] || badges.manual;
  };

  const getRequestTypeBadge = (type?: string) => {
    if (!type) return null;
    const badges = {
      response: <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">Response</span>,
      investigation: <span className="px-2 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400">Investigation</span>,
      automation: <span className="px-2 py-1 rounded-full text-xs bg-purple-500/20 text-purple-400">Automation</span>,
      reporting: <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">Reporting</span>,
      qa: <span className="px-2 py-1 rounded-full text-xs bg-gray-500/20 text-gray-400">Q&A</span>
    };
    return badges[type as keyof typeof badges];
  };

  const filteredTriggers = triggers.filter(t => {
    if (activeTab === "active" && t.status === "archived") return false;
    if (activeTab === "archived" && t.status !== "archived") return false;
    if (filterStatus !== "all" && t.status !== filterStatus) return false;
    if (filterSource !== "all" && t.source !== filterSource) return false;
    if (searchQuery && !t.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !t.description?.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading automation center...</p>
        </div>
      </div>
    );
  }

  return (
    <DashboardLayout breadcrumbs={[{ label: "Automations" }]}>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-gray-400 mt-1">
                Manage automated workflows, NLP suggestions, and trigger policies
              </p>
            </div>
          </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 border-b border-gray-800">
          {[
            { id: "active", label: "Active Automations", icon: Activity },
            { id: "suggestions", label: "NLP Suggestions", icon: Bot },
            { id: "archived", label: "Archived", icon: Archive },
            { id: "insights", label: "Coverage Insights", icon: TrendingUp }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === id
                  ? "border-blue-500 text-blue-400"
                  : "border-transparent text-gray-400 hover:text-gray-300"
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Active Automations Tab */}
      {activeTab === "active" && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="flex items-center gap-4 p-4 bg-gray-800/50 border border-gray-700/50 rounded-xl">
            <div className="flex items-center gap-2 flex-1">
              <Search className="w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search automations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 bg-transparent border-none text-white placeholder-gray-400 focus:outline-none"
              />
            </div>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm"
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="paused">Paused</option>
              <option value="error">Error</option>
            </select>
            <select
              value={filterSource}
              onChange={(e) => setFilterSource(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm"
            >
              <option value="all">All Sources</option>
              <option value="nlp">NLP</option>
              <option value="manual">Manual</option>
              <option value="template">Template</option>
              <option value="api">API</option>
            </select>
            {selectedItems.length > 0 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleBulkAction("pause")}
                  className="px-3 py-2 bg-yellow-600/20 border border-yellow-500/30 rounded-lg text-sm hover:bg-yellow-600/30"
                >
                  Pause Selected
                </button>
                <button
                  onClick={() => handleBulkAction("archive")}
                  className="px-3 py-2 bg-gray-600/20 border border-gray-500/30 rounded-lg text-sm hover:bg-gray-600/30"
                >
                  Archive Selected
                </button>
              </div>
            )}
          </div>

          {/* Triggers Table */}
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">
                    <input
                      type="checkbox"
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedItems(filteredTriggers.map(t => t.id));
                        } else {
                          setSelectedItems([]);
                        }
                      }}
                      checked={selectedItems.length === filteredTriggers.length && filteredTriggers.length > 0}
                      className="rounded"
                    />
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Trigger Name</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Source</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Type</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Execution</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Stats</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Last Run</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-gray-400">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700/50">
                {filteredTriggers.map((trigger) => (
                  <tr key={trigger.id} className="hover:bg-gray-700/30 transition-colors">
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedItems.includes(trigger.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedItems([...selectedItems, trigger.id]);
                          } else {
                            setSelectedItems(selectedItems.filter(id => id !== trigger.id));
                          }
                        }}
                        className="rounded"
                      />
                    </td>
                    <td className="px-4 py-3">{getStatusBadge(trigger.status, trigger.enabled)}</td>
                    <td className="px-4 py-3">
                      <div>
                        <div className="font-medium text-white">{trigger.name}</div>
                        {trigger.description && (
                          <div className="text-xs text-gray-400 truncate max-w-xs">{trigger.description}</div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">{getSourceBadge(trigger.source)}</td>
                    <td className="px-4 py-3">{getRequestTypeBadge(trigger.request_type)}</td>
                    <td className="px-4 py-3">
                      {trigger.auto_execute ? (
                        <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">Auto</span>
                      ) : (
                        <span className="px-2 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400">Manual</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm">
                        <div className="text-white">{trigger.trigger_count} runs</div>
                        <div className="text-xs text-gray-400">{Math.round(trigger.success_rate * 100)}% success</div>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {trigger.last_triggered_at ? (
                        <div className="text-sm text-gray-400">
                          {new Date(trigger.last_triggered_at).toLocaleString()}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-500">Never</div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => setSelectedTrigger(trigger)}
                          className="p-1 hover:bg-gray-600 rounded"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4 text-gray-400" />
                        </button>
                        <button
                          onClick={() => setSettingsTrigger(trigger)}
                          className="p-1 hover:bg-gray-600 rounded"
                          title="Settings"
                        >
                          <Settings className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 hover:bg-gray-600 rounded" title="Edit">
                          <Edit className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 hover:bg-gray-600 rounded" title="Delete">
                          <Trash2 className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* NLP Suggestions Tab */}
      {activeTab === "suggestions" && (
        <div className="space-y-6">
          <div className="text-sm text-gray-400 mb-4">
            {suggestions.filter(s => s.status === "pending").length} suggestions awaiting review
          </div>
          <div className="grid grid-cols-1 gap-4">
            {suggestions.filter(s => s.status === "pending").map((suggestion) => (
              <div key={suggestion.id} className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {getRequestTypeBadge(suggestion.request_type)}
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        suggestion.priority === "critical" ? "bg-red-500/20 text-red-400" :
                        suggestion.priority === "high" ? "bg-orange-500/20 text-orange-400" :
                        "bg-blue-500/20 text-blue-400"
                      }`}>
                        {suggestion.priority.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-400">
                        Confidence: {Math.round(suggestion.confidence * 100)}%
                      </span>
                    </div>
                    <div className="text-white font-medium mb-2">"{suggestion.prompt}"</div>
                    <div className="text-sm text-gray-400">
                      {suggestion.detected_actions?.length || 0} actions detected
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm transition-colors">
                    <CheckCircle className="w-4 h-4 inline mr-2" />
                    Approve & Automate
                  </button>
                  <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors">
                    Convert to Manual
                  </button>
                  <button className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm transition-colors">
                    <XCircle className="w-4 h-4 inline mr-2" />
                    Dismiss
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Archived Tab */}
      {activeTab === "archived" && (
        <div className="space-y-6">
          <div className="text-sm text-gray-400 mb-4">
            {filteredTriggers.length} archived automations
          </div>
          {/* Similar table structure to active, but read-only */}
          <div className="text-center text-gray-400 py-12">
            Archived automations will appear here
          </div>
        </div>
      )}

      {/* Coverage Insights Tab */}
      {activeTab === "insights" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">Parser Coverage</div>
              <div className="text-3xl font-bold text-white">87%</div>
              <div className="text-xs text-green-400 mt-2">+5% from last month</div>
            </div>
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">Avg Confidence</div>
              <div className="text-3xl font-bold text-white">0.82</div>
              <div className="text-xs text-blue-400 mt-2">Across all NLP triggers</div>
            </div>
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6">
              <div className="text-gray-400 text-sm mb-2">Fallback Rate</div>
              <div className="text-3xl font-bold text-white">12%</div>
              <div className="text-xs text-orange-400 mt-2">Target: &lt;10%</div>
            </div>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {selectedTrigger && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedTrigger(null)}>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">{selectedTrigger.name}</h2>
              <button onClick={() => setSelectedTrigger(null)} className="text-gray-400 hover:text-white">
                <XCircle className="w-6 h-6" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <div className="text-sm text-gray-400 mb-1">Description</div>
                <div className="text-white">{selectedTrigger.description || "No description"}</div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Source</div>
                  <div>{getSourceBadge(selectedTrigger.source)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Request Type</div>
                  <div>{getRequestTypeBadge(selectedTrigger.request_type) || "N/A"}</div>
                </div>
              </div>
              {selectedTrigger.source_prompt && (
                <div>
                  <div className="text-sm text-gray-400 mb-1">Original NLP Prompt</div>
                  <div className="text-white bg-gray-700/50 p-3 rounded-lg">"{selectedTrigger.source_prompt}"</div>
                </div>
              )}
              <div>
                <div className="text-sm text-gray-400 mb-1">Workflow Steps ({selectedTrigger.workflow_steps.length})</div>
                <div className="space-y-2">
                  {selectedTrigger.workflow_steps.map((step, idx) => (
                    <div key={idx} className="bg-gray-700/30 p-3 rounded-lg">
                      <div className="text-white font-medium">{idx + 1}. {step.action_type}</div>
                      <div className="text-sm text-gray-400">Category: {step.category}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {settingsTrigger && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSettingsTrigger(null)}>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 max-w-lg w-full" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <Settings className="w-6 h-6" />
                Trigger Settings
              </h2>
              <button onClick={() => setSettingsTrigger(null)} className="text-gray-400 hover:text-white">
                <XCircle className="w-6 h-6" />
              </button>
            </div>
            <div className="space-y-6">
              <div>
                <div className="text-sm font-medium text-gray-300 mb-2">{settingsTrigger.name}</div>
                <div className="text-xs text-gray-400">{settingsTrigger.description || "No description"}</div>
              </div>

              {/* Auto Execute Toggle */}
              <div className="bg-gray-700/30 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-400" />
                    <span className="font-medium text-white">Execution Mode</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settingsTrigger.auto_execute}
                      onChange={(e) => {
                        const newValue = e.target.checked;
                        setSettingsTrigger({ ...settingsTrigger, auto_execute: newValue });
                      }}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                <div className="text-sm text-gray-400">
                  {settingsTrigger.auto_execute ? (
                    <span className="text-red-400">⚡ Automatic - Workflows execute immediately when triggered</span>
                  ) : (
                    <span className="text-blue-400">✋ Manual Approval - Workflows require approval before execution</span>
                  )}
                </div>
              </div>

              {/* Priority Setting */}
              <div className="bg-gray-700/30 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                  <span className="font-medium text-white">Priority Level</span>
                </div>
                <select
                  value={settingsTrigger.priority}
                  onChange={(e) => setSettingsTrigger({ ...settingsTrigger, priority: e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
                <button
                  onClick={() => updateTriggerSettings(settingsTrigger.id, {
                    auto_execute: settingsTrigger.auto_execute,
                    priority: settingsTrigger.priority,
                    editor: "SOC Analyst"
                  })}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Save Changes
                </button>
                <button
                  onClick={() => setSettingsTrigger(null)}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      </div>
    </DashboardLayout>
  );
}
