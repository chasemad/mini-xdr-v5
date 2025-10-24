"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  CheckCircle,
  Loader2,
  AlertCircle,
  Cloud,
  Server,
  Shield,
  RefreshCw,
  ChevronRight,
  Clock,
} from "lucide-react";
import { ActionButton } from "../ui/ActionButton";
import { StatusChip } from "../ui/StatusChip";
import { onboardingV2API, ProgressResponse, CloudAsset } from "../../lib/api/onboardingV2";

interface OnboardingProgressProps {
  onComplete?: () => void;
}

export function OnboardingProgress({ onComplete }: OnboardingProgressProps) {
  const router = useRouter();
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [assets, setAssets] = useState<CloudAsset[]>([]);
  const [polling, setPolling] = useState(true);

  useEffect(() => {
    fetchProgress();

    // Poll for updates every 5 seconds
    const interval = setInterval(fetchProgress, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchProgress = async () => {
    try {
      setError("");
      const [progressData, assetsData] = await Promise.all([
        onboardingV2API.getProgress(),
        onboardingV2API.getAssets(),
      ]);

      setProgress(progressData);
      setAssets(assetsData.assets);

      // Stop polling if complete or error
      if (progressData.overall_status === 'completed' || progressData.overall_status === 'error') {
        setPolling(false);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'info';
      case 'error':
        return 'error';
      default:
        return 'pending';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      default:
        return 'pending';
    }
  };

  const handleContinue = () => {
    if (onComplete) {
      onComplete();
    } else {
      router.push("/");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading onboarding progress...</p>
        </div>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <p className="text-red-400 mb-4">Failed to load progress</p>
          <ActionButton onClick={fetchProgress} icon={<RefreshCw className="w-4 h-4" />}>
            Retry
          </ActionButton>
        </div>
      </div>
    );
  }

  const isComplete = progress.overall_status === 'completed';
  const hasError = progress.overall_status === 'error';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Onboarding Progress</h1>
          <p className="text-gray-400">
            Auto-discovery and agent deployment in progress
            {polling && <span className="ml-2 text-blue-400">• Live updates</span>}
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-200">{error}</span>
          </div>
        )}

        {/* Progress Overview */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {/* Discovery */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Cloud className="w-6 h-6 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">Discovery</h3>
              </div>
              {getStatusIcon(progress.discovery.status)}
            </div>

            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>{Math.round(progress.discovery.progress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.discovery.progress}%` }}
                />
              </div>
            </div>

            <div className="text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Assets Found:</span>
                <span className="font-semibold">{progress.discovery.assets_found}</span>
              </div>
            </div>

            {progress.discovery.message && (
              <p className="text-xs text-gray-500 mt-2">{progress.discovery.message}</p>
            )}
          </div>

          {/* Deployment */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Server className="w-6 h-6 text-green-400" />
                <h3 className="text-lg font-semibold text-white">Deployment</h3>
              </div>
              {getStatusIcon(progress.deployment.status)}
            </div>

            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>{Math.round(progress.deployment.progress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.deployment.progress}%` }}
                />
              </div>
            </div>

            <div className="text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Agents Deployed:</span>
                <span className="font-semibold">
                  {progress.deployment.agents_deployed}/{progress.deployment.total_assets}
                </span>
              </div>
            </div>

            {progress.deployment.message && (
              <p className="text-xs text-gray-500 mt-2">{progress.deployment.message}</p>
            )}
          </div>

          {/* Validation */}
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Shield className="w-6 h-6 text-purple-400" />
                <h3 className="text-lg font-semibold text-white">Validation</h3>
              </div>
              {getStatusIcon(progress.validation.status)}
            </div>

            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>{Math.round(progress.validation.progress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.validation.progress}%` }}
                />
              </div>
            </div>

            <div className="text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Checks Passed:</span>
                <span className="font-semibold">
                  {progress.validation.checks_passed}/{progress.validation.total_checks}
                </span>
              </div>
            </div>

            {progress.validation.message && (
              <p className="text-xs text-gray-500 mt-2">{progress.validation.message}</p>
            )}
          </div>
        </div>

        {/* Assets Overview */}
        {assets.length > 0 && (
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 mb-8">
            <h3 className="text-xl font-semibold text-white mb-6">
              Discovered Assets ({assets.length})
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs uppercase bg-gray-900 text-gray-400">
                  <tr>
                    <th className="px-4 py-3">Asset</th>
                    <th className="px-4 py-3">Type</th>
                    <th className="px-4 py-3">Region</th>
                    <th className="px-4 py-3">Agent Status</th>
                    <th className="px-4 py-3">Priority</th>
                  </tr>
                </thead>
                <tbody>
                  {assets.slice(0, 10).map((asset) => (
                    <tr key={asset.id} className="border-b border-gray-700">
                      <td className="px-4 py-3">
                        <div>
                          <div className="text-white font-medium">
                            {asset.hostname || asset.asset_id}
                          </div>
                          <div className="text-xs text-gray-500">
                            {asset.ip_address}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-gray-300 capitalize">
                        {asset.asset_type}
                      </td>
                      <td className="px-4 py-3 text-gray-300">
                        {asset.region}
                      </td>
                      <td className="px-4 py-3">
                        <StatusChip status={asset.agent_status as any} />
                      </td>
                      <td className="px-4 py-3">
                        <StatusChip
                          status={getPriorityColor(asset.deployment_priority) as any}
                          label={asset.deployment_priority}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {assets.length > 10 && (
              <p className="text-sm text-gray-500 mt-4 text-center">
                Showing first 10 of {assets.length} assets
              </p>
            )}
          </div>
        )}

        {/* Completion Actions */}
        <div className="text-center">
          {isComplete ? (
            <div className="space-y-4">
              <div className="inline-flex items-center gap-3 px-6 py-3 bg-green-900/20 border border-green-800 rounded-lg">
                <CheckCircle className="w-6 h-6 text-green-400" />
                <span className="text-green-200 font-semibold">
                  Onboarding Complete! All systems secured.
                </span>
              </div>

              <ActionButton
                onClick={handleContinue}
                icon={<ChevronRight className="w-5 h-5" />}
                className="px-8"
              >
                Go to Dashboard
              </ActionButton>
            </div>
          ) : hasError ? (
            <div className="space-y-4">
              <div className="inline-flex items-center gap-3 px-6 py-3 bg-red-900/20 border border-red-800 rounded-lg">
                <AlertCircle className="w-6 h-6 text-red-400" />
                <span className="text-red-200 font-semibold">
                  Onboarding encountered errors. Please check the logs.
                </span>
              </div>

              <div className="flex gap-4 justify-center">
                <ActionButton
                  onClick={fetchProgress}
                  variant="secondary"
                  icon={<RefreshCw className="w-4 h-4" />}
                >
                  Refresh Status
                </ActionButton>

                <ActionButton
                  onClick={handleContinue}
                  variant="outline"
                >
                  Continue Anyway
                </ActionButton>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="inline-flex items-center gap-3 px-6 py-3 bg-blue-900/20 border border-blue-800 rounded-lg">
                <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
                <span className="text-blue-200">
                  Onboarding in progress... This may take a few minutes.
                </span>
              </div>

              <ActionButton
                onClick={fetchProgress}
                variant="secondary"
                icon={<RefreshCw className="w-4 h-4" />}
              >
                Refresh Status
              </ActionButton>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
