"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2, AlertCircle } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { QuickStartOnboarding } from "../components/onboarding/QuickStartOnboarding";
import { OnboardingProgress } from "../components/onboarding/OnboardingProgress";
import { ActionButton } from "../../components/ui/ActionButton";

export default function OnboardingPage() {
  const router = useRouter();
  const { user, organization } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [onboardingFlow, setOnboardingFlow] = useState<string | null>(null);
  const [showProgress, setShowProgress] = useState(false);

  useEffect(() => {
    checkOnboardingFlow();
  }, [organization]);

  const checkOnboardingFlow = async () => {
    try {
      if (!organization) {
        setError("No organization found. Please log in again.");
        setLoading(false);
        return;
      }

      // Check URL parameter to see if we should show progress
      const urlParams = new URLSearchParams(window.location.search);
      const showProgressParam = urlParams.get('progress');

      if (showProgressParam === 'true') {
        setShowProgress(true);
        setOnboardingFlow('seamless'); // Assume seamless if showing progress
      } else {
        // Check organization's onboarding flow version
        const flowVersion = (organization as any).onboarding_flow_version || 'legacy';
        setOnboardingFlow(flowVersion);
      }

      setLoading(false);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleSeamlessComplete = () => {
    router.push("/");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading onboarding...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Onboarding Error</h2>
          <p className="text-red-200 mb-4">{error}</p>
          <ActionButton onClick={() => router.push("/")}>
            Go to Dashboard
          </ActionButton>
        </div>
      </div>
    );
  }

  // Show seamless onboarding for organizations with seamless flow
  if (onboardingFlow === 'seamless') {
    if (showProgress) {
      return <OnboardingProgress onComplete={handleSeamlessComplete} />;
    } else {
      return <QuickStartOnboarding onComplete={handleSeamlessComplete} />;
    }
  }

  // Fallback to legacy onboarding (for now, just show a message)
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-8 px-4 flex items-center justify-center">
      <div className="text-center max-w-2xl">
        <AlertCircle className="w-16 h-16 text-yellow-400 mx-auto mb-6" />
        <h1 className="text-3xl font-bold text-white mb-4">Legacy Onboarding</h1>
        <p className="text-gray-400 mb-6">
          This organization is configured to use the legacy onboarding flow, which is currently under maintenance.
          Please contact your administrator to upgrade to seamless onboarding.
        </p>
        <div className="space-y-3">
          <ActionButton onClick={() => router.push("/")}>
            Go to Dashboard
          </ActionButton>
          <p className="text-sm text-gray-500">
            Organization: {organization?.name} (Flow: {onboardingFlow})
          </p>
        </div>
      </div>
    </div>
  );
}