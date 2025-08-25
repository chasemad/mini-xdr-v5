"use client";

import { useEffect, useState } from "react";
import { getHealth, getAutoContainSetting, setAutoContainSetting } from "../lib/api";

interface HealthData {
  status: string;
  timestamp: string;
  auto_contain: boolean;
}

interface AutoContainData {
  enabled: boolean;
}

export default function SystemStatus() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [autoContain, setAutoContain] = useState<AutoContainData | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);

  const fetchStatus = async () => {
    try {
      const [healthData, autoContainData] = await Promise.all([
        getHealth(),
        getAutoContainSetting(),
      ]);
      
      setHealth(healthData);
      setAutoContain(autoContainData);
    } catch (error) {
      console.error("Failed to fetch system status:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAutoContainToggle = async () => {
    if (!autoContain || toggling) return;

    setToggling(true);
    try {
      const newSetting = await setAutoContainSetting(!autoContain.enabled);
      setAutoContain(newSetting);
    } catch (error) {
      console.error("Failed to toggle auto-contain:", error);
    } finally {
      setToggling(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-20 bg-gray-200 rounded-lg"></div>
            <div className="h-20 bg-gray-200 rounded-lg"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">System Status</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Health */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-700">Health</h3>
            <div className={`h-3 w-3 rounded-full ${
              health?.status === "healthy" ? "bg-green-500" : "bg-red-500"
            }`}></div>
          </div>
          <p className={`text-lg font-semibold capitalize ${
            health?.status === "healthy" ? "text-green-700" : "text-red-700"
          }`}>
            {health?.status || "Unknown"}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Last check: {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : "Unknown"}
          </p>
        </div>

        {/* Auto-Contain Setting */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-700">Auto-Contain</h3>
            <button
              onClick={handleAutoContainToggle}
              disabled={toggling}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:ring-offset-2 ${
                autoContain?.enabled ? "bg-indigo-600" : "bg-gray-200"
              } ${toggling ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                  autoContain?.enabled ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
          <p className={`text-lg font-semibold ${
            autoContain?.enabled ? "text-indigo-700" : "text-gray-600"
          }`}>
            {autoContain?.enabled ? "Enabled" : "Disabled"}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Automatic threat response
          </p>
        </div>
      </div>

      {/* Network Environment */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Network Environment</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="flex items-center space-x-2">
            <div className="h-2 w-2 rounded-full bg-green-500"></div>
            <span className="text-xs text-gray-600">Host Mac (10.0.0.123)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="h-2 w-2 rounded-full bg-green-500"></div>
            <span className="text-xs text-gray-600">Honeypot (10.0.0.23)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
            <span className="text-xs text-gray-600">Kali (10.0.0.182)</span>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Detection: SSH Brute-Force • Response: UFW Block/Unblock
        </p>
      </div>
    </div>
  );
}
