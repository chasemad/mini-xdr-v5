"use client";

import { useEffect, useState } from "react";
import { getIncidents } from "./lib/api";
import RealTimeIncidents from "./components/RealTimeIncidents";
import BlockedIPs from "./components/BlockedIPs";
import SystemStatus from "./components/SystemStatus";

export default function Overview() {
  const [stats, setStats] = useState({
    totalIncidents: 0,
    activeIncidents: 0,
    blockedIPs: 0,
    lastActivity: null as Date | null
  });
  const [loading, setLoading] = useState(true);

  const fetchStats = async () => {
    try {
      const incidents = await getIncidents();
      
      const activeIncidents = incidents.filter(i => i.status === "open").length;
      const blockedIPs = incidents.filter(i => i.status === "contained").length;
      const lastActivity = incidents.length > 0 ? new Date(incidents[0].created_at) : null;
      
      setStats({
        totalIncidents: incidents.length,
        activeIncidents,
        blockedIPs,
        lastActivity
      });
    } catch (error) {
      console.error("Failed to fetch stats:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    
    // Refresh stats every 5 seconds for more real-time updates
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="px-4 py-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid gap-6 lg:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-2xl"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-6 space-y-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Security Operations Center</h1>
        <p className="mt-2 text-gray-600">
          Real-time threat detection and response dashboard
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-6 lg:grid-cols-4">
        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-700">Total Incidents</h2>
            <div className="h-8 w-8 bg-blue-100 rounded-lg flex items-center justify-center">
              <svg className="h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-2xl font-bold text-gray-900">{stats.totalIncidents}</p>
            <p className="text-sm text-gray-600">All time</p>
          </div>
        </div>

        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-700">Active Incidents</h2>
            <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${
              stats.activeIncidents > 0 ? 'bg-yellow-100' : 'bg-green-100'
            }`}>
              <svg className={`h-4 w-4 ${
                stats.activeIncidents > 0 ? 'text-yellow-600' : 'text-green-600'
              }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-2xl font-bold text-gray-900">{stats.activeIncidents}</p>
            <p className="text-sm text-gray-600">Requiring attention</p>
          </div>
        </div>

        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-700">Blocked IPs</h2>
            <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${
              stats.blockedIPs > 0 ? 'bg-red-100' : 'bg-green-100'
            }`}>
              <svg className={`h-4 w-4 ${
                stats.blockedIPs > 0 ? 'text-red-600' : 'text-green-600'
              }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L5.636 5.636" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-2xl font-bold text-gray-900">{stats.blockedIPs}</p>
            <p className="text-sm text-gray-600">Currently contained</p>
          </div>
        </div>

        <div className="p-6 rounded-2xl shadow-sm border border-gray-200 bg-white">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-700">Last Activity</h2>
            <div className="h-8 w-8 bg-purple-100 rounded-lg flex items-center justify-center">
              <svg className="h-4 w-4 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div className="mt-4">
            <p className="text-sm font-bold text-gray-900">
              {stats.lastActivity ? stats.lastActivity.toLocaleTimeString() : "None"}
            </p>
            <p className="text-sm text-gray-600">
              {stats.lastActivity ? stats.lastActivity.toLocaleDateString() : "No recent activity"}
            </p>
          </div>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid gap-8 lg:grid-cols-3">
        {/* Left Column - System Status */}
        <div className="lg:col-span-1">
          <SystemStatus />
        </div>
        
        {/* Center Column - Real-time Incidents */}
        <div className="lg:col-span-1">
          <RealTimeIncidents />
        </div>
        
        {/* Right Column - Blocked IPs */}
        <div className="lg:col-span-1">
          <BlockedIPs />
        </div>
      </div>

      {/* Alert Banner for Critical Status */}
      {stats.activeIncidents > 3 && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                High Alert: Multiple Active Incidents
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>
                  {stats.activeIncidents} incidents require immediate attention. 
                  Consider enabling auto-contain or manually reviewing threats.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
