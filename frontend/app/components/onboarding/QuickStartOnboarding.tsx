"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Cloud,
  Key,
  Shield,
  Zap,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { ActionButton } from "../../components/ui/ActionButton";
import { StatusChip } from "../../components/ui/StatusChip";
import { onboardingV2API, CloudCredentials } from "../../../lib/api/onboardingV2";

interface QuickStartOnboardingProps {
  onComplete?: () => void;
}

export function QuickStartOnboarding({ onComplete }: QuickStartOnboardingProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const [credentials, setCredentials] = useState<CloudCredentials>({
    role_arn: "",
    external_id: "mini-xdr-external-id", // Default for demo
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await onboardingV2API.quickStart({
        provider: "aws",
        credentials,
      });

      setSuccess(true);

      // Redirect to progress page after a short delay
      setTimeout(() => {
        router.push("/onboarding/progress");
      }, 2000);

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSkip = () => {
    router.push("/");
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-900 via-gray-900 to-black py-8 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-12">
            <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-6" />
            <h1 className="text-3xl font-bold text-white mb-4">Onboarding Started!</h1>
            <p className="text-gray-400 mb-8">
              Auto-discovery and agent deployment have begun. You'll be redirected to track progress shortly.
            </p>
            <div className="flex justify-center">
              <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Shield className="w-12 h-12 text-blue-400 mr-4" />
            <h1 className="text-4xl font-bold text-white">Mini-XDR</h1>
          </div>
          <h2 className="text-2xl font-semibold text-gray-200 mb-2">
            Seamless Cloud Onboarding
          </h2>
          <p className="text-gray-400">
            Connect your AWS account and we'll automatically discover and secure your infrastructure
          </p>
        </div>

        {/* Benefits */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center">
            <Zap className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">5-Minute Setup</h3>
            <p className="text-gray-400 text-sm">
              No manual configuration or network scanning required
            </p>
          </div>

          <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center">
            <Cloud className="w-12 h-12 text-blue-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Cloud-Native</h3>
            <p className="text-gray-400 text-sm">
              Direct integration with AWS APIs for comprehensive asset discovery
            </p>
          </div>

          <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center">
            <Shield className="w-12 h-12 text-green-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Automated Security</h3>
            <p className="text-gray-400 text-sm">
              Intelligent agent deployment with priority-based protection
            </p>
          </div>
        </div>

        {/* Main Form */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
          <div className="max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold text-white mb-6 text-center">
              Connect AWS Account
            </h3>

            {error && (
              <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-red-200">{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  IAM Role ARN
                </label>
                <div className="relative">
                  <Key className="absolute left-3 top-3 w-5 h-5 text-gray-500" />
                  <input
                    type="text"
                    value={credentials.role_arn}
                    onChange={(e) => setCredentials({ ...credentials, role_arn: e.target.value })}
                    placeholder="arn:aws:iam::123456789012:role/MiniXDRRole"
                    className="w-full pl-10 pr-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  The IAM role ARN with permissions to discover EC2 and RDS resources
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  External ID
                </label>
                <input
                  type="text"
                  value={credentials.external_id}
                  onChange={(e) => setCredentials({ ...credentials, external_id: e.target.value })}
                  placeholder="mini-xdr-external-id"
                  className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">
                  Trust policy identifier for secure cross-account access
                </p>
              </div>

              <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-blue-200 mb-2">What happens next?</h4>
                <ul className="text-sm text-blue-100 space-y-1">
                  <li>• We'll scan all your AWS regions for EC2 instances and RDS databases</li>
                  <li>• Priority-based agent deployment (critical systems first)</li>
                  <li>• Real-time progress tracking with automatic validation</li>
                  <li>• Complete security coverage in under 5 minutes</li>
                </ul>
              </div>

              <div className="flex gap-4">
                <ActionButton
                  type="submit"
                  loading={loading}
                  icon={<Cloud className="w-5 h-5" />}
                  className="flex-1"
                >
                  {loading ? "Connecting..." : "Connect AWS Account"}
                </ActionButton>

                <ActionButton
                  type="button"
                  variant="secondary"
                  onClick={handleSkip}
                  className="px-8"
                >
                  Skip
                </ActionButton>
              </div>
            </form>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-gray-500 text-sm">
            Need help setting up IAM permissions?{" "}
            <a
              href="/docs/aws-iam-setup"
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              View setup guide
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
