"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  CheckCircle,
  Circle,
  Loader2,
  ChevronRight,
  AlertCircle,
  Copy,
  Check,
  RefreshCw,
} from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { ActionButton } from "../../components/ui/ActionButton";
import { StatusChip } from "../../components/ui/StatusChip";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Step {
  id: string;
  title: string;
  description: string;
}

const steps: Step[] = [
  {
    id: "profile",
    title: "Organization Profile",
    description: "Basic information about your organization",
  },
  {
    id: "network_scan",
    title: "Network Discovery",
    description: "Scan your network to discover assets",
  },
  {
    id: "agents",
    title: "Agent Deployment",
    description: "Deploy security agents to your systems",
  },
  {
    id: "permissions",
    title: "Permissions & Approval",
    description: "Grant action rights and provide credentials",
  },
  {
    id: "validation",
    title: "Validation",
    description: "Verify setup and configuration",
  },
];

export default function OnboardingPage() {
  const router = useRouter();
  const { user, organization } = useAuth();
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Step 1: Profile
  const [profile, setProfile] = useState({
    region: "",
    industry: "",
    company_size: "small",
  });

  // Step 2: Network scan
  const [networkRanges, setNetworkRanges] = useState<string[]>([""]);
  const [scanning, setScanning] = useState(false);
  const [scanResults, setScanResults] = useState<any[]>([]);
  const [scanId, setScanId] = useState("");

  // Step 3: Agents
  const [selectedPlatform, setSelectedPlatform] = useState<string>("linux");
  const [agentTokens, setAgentTokens] = useState<any[]>([]);
  const [enrolledAgents, setEnrolledAgents] = useState<any[]>([]);
  const [copiedToken, setCopiedToken] = useState<string | null>(null);

  // Step 4: Permissions & Approval
  const [allowActions, setAllowActions] = useState<boolean>(false);
  const [credentials, setCredentials] = useState<Array<{ key: string; value: string; scope: string }>>([
    { key: "", value: "", scope: "linux" },
  ]);

  // Step 5: Validation
  const [validationChecks, setValidationChecks] = useState<any[]>([]);
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    initializeOnboarding();
  }, []);

  useEffect(() => {
    // Poll for enrolled agents when on agents step
    if (currentStep === 2) {
      const interval = setInterval(fetchEnrolledAgents, 5000);
      return () => clearInterval(interval);
    }
  }, [currentStep]);

  const initializeOnboarding = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/start`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to start onboarding");

      const data = await response.json();
      
      // Set step based on backend state
      const stepMap: Record<string, number> = {
        profile: 0,
        network_scan: 1,
        agents: 2,
        validation: 3,
      };
      
      if (data.onboarding_step && stepMap[data.onboarding_step] !== undefined) {
        setCurrentStep(stepMap[data.onboarding_step]);
      }
    } catch (err: any) {
      setError(err.message);
    }
  };

  const saveProfile = async () => {
    setLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/profile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) throw new Error("Failed to save profile");

      setCurrentStep(1);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const startNetworkScan = async () => {
    setScanning(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/network-scan`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          network_ranges: networkRanges.filter((r) => r.trim()),
          scan_type: "quick",
        }),
      });

      if (!response.ok) throw new Error("Network scan failed");

      const data = await response.json();
      setScanId(data.scan_id);

      // Fetch results
      await fetchScanResults(data.scan_id);
      
      setCurrentStep(2);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setScanning(false);
    }
  };

  const fetchScanResults = async (sid?: string) => {
    try {
      const token = localStorage.getItem("access_token");
      const url = sid
        ? `${API_BASE}/api/onboarding/scan-results?scan_id=${sid}`
        : `${API_BASE}/api/onboarding/scan-results`;
      
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to fetch scan results");

      const data = await response.json();
      setScanResults(data);
    } catch (err: any) {
      console.error("Error fetching scan results:", err);
    }
  };

  const generateAgentToken = async () => {
    setLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/generate-agent-token`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          platform: selectedPlatform,
        }),
      });

      if (!response.ok) throw new Error("Failed to generate token");

      const data = await response.json();
      setAgentTokens([...agentTokens, data]);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchEnrolledAgents = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/enrolled-agents`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) return;

      const data = await response.json();
      setEnrolledAgents(data);
    } catch (err) {
      console.error("Error fetching enrolled agents:", err);
    }
  };

  const savePermissions = async () => {
    setLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/permissions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          allow_actions: allowActions,
          credentials: credentials
            .filter((c) => c.key && c.value)
            .map((c) => ({ key: c.key, value: c.value, scope: c.scope })),
        }),
      });

      if (!response.ok) throw new Error("Failed to save permissions");

      setCurrentStep(4); // go to validation
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runValidation = async () => {
    setValidating(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/validation`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Validation failed");

      const data = await response.json();
      setValidationChecks(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setValidating(false);
    }
  };

  const completeOnboarding = async () => {
    setLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_BASE}/api/onboarding/complete`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to complete onboarding");

      // Redirect to dashboard
      router.push("/");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedToken(id);
    setTimeout(() => setCopiedToken(null), 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Welcome to Mini-XDR</h1>
          <p className="text-gray-400">Let's get your security operations center set up</p>
        </div>

        {/* Stepper */}
        <div className="mb-12">
          <div className="flex items-center justify-between max-w-3xl mx-auto">
            {steps.map((step, index) => (
              <React.Fragment key={step.id}>
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`
                      w-10 h-10 rounded-full flex items-center justify-center border-2 mb-2
                      ${
                        index < currentStep
                          ? "bg-green-600 border-green-600"
                          : index === currentStep
                          ? "bg-blue-600 border-blue-600"
                          : "bg-gray-800 border-gray-700"
                      }
                    `}
                  >
                    {index < currentStep ? (
                      <CheckCircle className="w-5 h-5 text-white" />
                    ) : (
                      <Circle className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-medium text-white">{step.title}</div>
                    <div className="text-xs text-gray-500 hidden sm:block">
                      {step.description}
                    </div>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`
                      h-0.5 flex-1 mx-4 mt-5
                      ${index < currentStep ? "bg-green-600" : "bg-gray-700"}
                    `}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-200">{error}</span>
          </div>
        )}

        {/* Step content */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
          {/* Step 1: Profile */}
          {currentStep === 0 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6">Organization Profile</h2>
              <div className="space-y-4 max-w-2xl">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Organization Name
                  </label>
                  <input
                    type="text"
                    value={organization?.name || ""}
                    disabled
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Region
                  </label>
                  <select
                    value={profile.region}
                    onChange={(e) => setProfile({ ...profile, region: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="">Select region</option>
                    <option value="us-east">US East</option>
                    <option value="us-west">US West</option>
                    <option value="eu">Europe</option>
                    <option value="apac">Asia Pacific</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Industry
                  </label>
                  <select
                    value={profile.industry}
                    onChange={(e) => setProfile({ ...profile, industry: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="">Select industry</option>
                    <option value="technology">Technology</option>
                    <option value="finance">Finance</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="retail">Retail</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Company Size
                  </label>
                  <select
                    value={profile.company_size}
                    onChange={(e) => setProfile({ ...profile, company_size: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="small">1-50 employees</option>
                    <option value="medium">51-500 employees</option>
                    <option value="large">501-5000 employees</option>
                    <option value="enterprise">5000+ employees</option>
                  </select>
                </div>

                <div className="pt-4">
                  <ActionButton
                    onClick={saveProfile}
                    loading={loading}
                    icon={<ChevronRight className="w-4 h-4" />}
                  >
                    Continue to Network Discovery
                  </ActionButton>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Network Scan */}
          {currentStep === 1 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6">Network Discovery</h2>
              <p className="text-gray-400 mb-6">
                Enter your network CIDR ranges to discover assets (e.g., 10.0.0.0/24, 192.168.1.0/24)
              </p>

              <div className="space-y-4 max-w-2xl mb-6">
                {networkRanges.map((range, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="text"
                      value={range}
                      onChange={(e) => {
                        const newRanges = [...networkRanges];
                        newRanges[index] = e.target.value;
                        setNetworkRanges(newRanges);
                      }}
                      placeholder="10.0.0.0/24"
                      className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                    />
                    {networkRanges.length > 1 && (
                      <ActionButton
                        variant="danger"
                        onClick={() => {
                          setNetworkRanges(networkRanges.filter((_, i) => i !== index));
                        }}
                      >
                        Remove
                      </ActionButton>
                    )}
                  </div>
                ))}

                <ActionButton
                  variant="secondary"
                  onClick={() => setNetworkRanges([...networkRanges, ""])}
                >
                  Add Another Range
                </ActionButton>
              </div>

              {scanResults.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Discovered Assets ({scanResults.length})
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs uppercase bg-gray-900 text-gray-400">
                        <tr>
                          <th className="px-4 py-3">IP Address</th>
                          <th className="px-4 py-3">Hostname</th>
                          <th className="px-4 py-3">OS Type</th>
                          <th className="px-4 py-3">Classification</th>
                          <th className="px-4 py-3">Priority</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scanResults.slice(0, 10).map((asset) => (
                          <tr key={asset.id} className="border-b border-gray-700">
                            <td className="px-4 py-3 font-mono text-blue-400">{asset.ip}</td>
                            <td className="px-4 py-3 text-gray-300">
                              {asset.hostname || "-"}
                            </td>
                            <td className="px-4 py-3 text-gray-300">{asset.os_type}</td>
                            <td className="px-4 py-3 text-gray-300">{asset.classification}</td>
                            <td className="px-4 py-3">
                              <StatusChip
                                status={asset.deployment_priority === "critical" ? "error" : "pending"}
                                label={asset.deployment_priority}
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="flex gap-3">
                <ActionButton
                  onClick={startNetworkScan}
                  loading={scanning}
                  disabled={!networkRanges.some((r) => r.trim())}
                >
                  {scanResults.length > 0 ? "Re-scan Network" : "Start Network Scan"}
                </ActionButton>

                {scanResults.length > 0 && (
                  <ActionButton
                    variant="secondary"
                    onClick={() => setCurrentStep(2)}
                    icon={<ChevronRight className="w-4 h-4" />}
                  >
                    Continue to Agent Deployment
                  </ActionButton>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Agents */}
          {currentStep === 2 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6">Agent Deployment</h2>
              <p className="text-gray-400 mb-6">
                Generate enrollment tokens and deploy agents to your systems
              </p>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Select Platform
                </label>
                <div className="flex gap-3">
                  {["linux", "windows", "macos", "docker"].map((platform) => (
                    <button
                      key={platform}
                      onClick={() => setSelectedPlatform(platform)}
                      className={`
                        px-4 py-2 rounded-lg border font-medium capitalize transition-colors
                        ${
                          selectedPlatform === platform
                            ? "bg-blue-600 border-blue-600 text-white"
                            : "bg-gray-700 border-gray-600 text-gray-300 hover:bg-gray-600"
                        }
                      `}
                    >
                      {platform}
                    </button>
                  ))}
                </div>
              </div>

              <ActionButton
                onClick={generateAgentToken}
                loading={loading}
                className="mb-6"
              >
                Generate {selectedPlatform} Agent Token
              </ActionButton>

              {agentTokens.length > 0 && (
                <div className="space-y-4 mb-6">
                  {agentTokens.map((token) => (
                    <div
                      key={token.enrollment_id}
                      className="p-4 bg-gray-900/50 border border-gray-700 rounded-lg"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="text-sm font-semibold text-white capitalize">
                            {token.platform} Agent
                          </div>
                          <div className="text-xs text-gray-500">
                            ID: {token.enrollment_id}
                          </div>
                        </div>
                        <StatusChip status={token.status as any} />
                      </div>

                      <div className="mb-3">
                        <div className="text-xs text-gray-500 mb-1">Enrollment Token</div>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 px-3 py-2 bg-gray-800 rounded text-xs text-green-400 font-mono overflow-x-auto">
                            {token.agent_token}
                          </code>
                          <button
                            onClick={() => copyToClipboard(token.agent_token, `token-${token.enrollment_id}`)}
                            className="p-2 text-gray-400 hover:text-white transition-colors"
                          >
                            {copiedToken === `token-${token.enrollment_id}` ? (
                              <Check className="w-4 h-4 text-green-400" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </div>

                      {token.install_scripts && (
                        <details className="text-sm">
                          <summary className="cursor-pointer text-blue-400 hover:text-blue-300">
                            View Install Script
                          </summary>
                          <pre className="mt-2 p-3 bg-gray-800 rounded text-xs text-gray-300 overflow-x-auto">
                            {token.install_scripts.bash || token.install_scripts.powershell || token.install_scripts.docker}
                          </pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {enrolledAgents.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Enrolled Agents ({enrolledAgents.length})
                  </h3>
                  <div className="space-y-2">
                    {enrolledAgents.map((agent) => (
                      <div
                        key={agent.enrollment_id}
                        className="flex items-center justify-between p-3 bg-gray-900/50 border border-gray-700 rounded-lg"
                      >
                        <div>
                          <div className="text-sm font-medium text-white">
                            {agent.hostname || agent.agent_id || "Unknown"}
                          </div>
                          <div className="text-xs text-gray-500 capitalize">
                            {agent.platform} • {agent.ip_address}
                          </div>
                        </div>
                        <StatusChip status={agent.status as any} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <ActionButton
                variant="secondary"
                onClick={() => setCurrentStep(3)}
                icon={<ChevronRight className="w-4 h-4" />}
                disabled={enrolledAgents.length === 0}
              >
                Continue to Validation
              </ActionButton>
            </div>
          )}

          {/* Step 4: Permissions & Approval */}
          {currentStep === 3 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6">Permissions & Approval</h2>
              <p className="text-gray-400 mb-6">Allow agents to take actions and provide optional credentials for privileged operations. Stored securely in a Kubernetes Secret.</p>

              <div className="mb-6">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={allowActions}
                    onChange={(e) => setAllowActions(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-gray-200">Allow agents to take remediation actions</span>
                </label>
              </div>

              <div className="space-y-3 mb-6 max-w-3xl">
                {credentials.map((c, idx) => (
                  <div key={idx} className="grid grid-cols-12 gap-2">
                    <select
                      value={c.scope}
                      onChange={(e) => {
                        const next = [...credentials];
                        next[idx].scope = e.target.value;
                        setCredentials(next);
                      }}
                      className="col-span-2 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm capitalize"
                    >
                      {['linux','windows','network','other'].map(s => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                    <input
                      placeholder="key (e.g., sudo_password)"
                      value={c.key}
                      onChange={(e) => {
                        const next = [...credentials];
                        next[idx].key = e.target.value;
                        setCredentials(next);
                      }}
                      className="col-span-4 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                    />
                    <input
                      placeholder="value"
                      value={c.value}
                      onChange={(e) => {
                        const next = [...credentials];
                        next[idx].value = e.target.value;
                        setCredentials(next);
                      }}
                      className="col-span-5 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                      type="password"
                    />
                    <button
                      className="col-span-1 px-3 py-2 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 rounded-lg text-sm text-red-300"
                      onClick={() => setCredentials(credentials.filter((_, i) => i !== idx))}
                    >
                      Remove
                    </button>
                  </div>
                ))}
                <button
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm"
                  onClick={() => setCredentials([...credentials, { key: "", value: "", scope: "linux" }])}
                >
                  Add credential
                </button>
              </div>

              <ActionButton onClick={savePermissions} loading={loading} icon={<ChevronRight className="w-4 h-4" />}>
                Save & Continue to Validation
              </ActionButton>
            </div>
          )}

          {/* Step 5: Validation */}
          {currentStep === 4 && (
            <div>
              <h2 className="text-2xl font-bold text-white mb-6">Validation</h2>
              <p className="text-gray-400 mb-6">
                Let's verify that everything is set up correctly
              </p>

              {validationChecks.length === 0 ? (
                <ActionButton
                  onClick={runValidation}
                  loading={validating}
                  icon={<RefreshCw className="w-4 h-4" />}
                >
                  Run Validation Checks
                </ActionButton>
              ) : (
                <div className="space-y-4 mb-6">
                  {validationChecks.map((check, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-4 p-4 bg-gray-900/50 border border-gray-700 rounded-lg"
                    >
                      <div className="mt-1">
                        {check.status === "pass" ? (
                          <CheckCircle className="w-5 h-5 text-green-400" />
                        ) : check.status === "fail" ? (
                          <AlertCircle className="w-5 h-5 text-red-400" />
                        ) : (
                          <Loader2 className="w-5 h-5 text-yellow-400 animate-spin" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-semibold text-white mb-1">
                          {check.check_name}
                        </div>
                        <div className="text-sm text-gray-400">{check.message}</div>
                      </div>
                    </div>
                  ))}

                  <div className="flex gap-3 pt-4">
                    <ActionButton
                      onClick={runValidation}
                      loading={validating}
                      variant="secondary"
                      icon={<RefreshCw className="w-4 h-4" />}
                    >
                      Re-run Checks
                    </ActionButton>

                    <ActionButton
                      onClick={completeOnboarding}
                      loading={loading}
                      disabled={validationChecks.some((c) => c.status === "fail")}
                    >
                      Complete Setup & Go to Dashboard
                    </ActionButton>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Skip option */}
        <div className="text-center mt-6">
          <button
            onClick={() => router.push("/api/onboarding/skip").then(() => router.push("/"))}
            className="text-sm text-gray-500 hover:text-gray-400 transition-colors"
          >
            Skip setup for now
          </button>
        </div>
      </div>
    </div>
  );
}


