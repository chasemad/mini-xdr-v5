"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { Shield, Server, Network, Zap, CheckCircle2 } from "lucide-react";
import { ActionButton } from "../../../components/ui/ActionButton";

const highlights = [
  {
    title: "Local-Only Stack",
    description:
      "PostgreSQL, Redis, backend, and frontend run together via Docker Compose with no cloud dependencies.",
    icon: Server,
    color: "text-blue-400",
  },
  {
    title: "Local ML Models",
    description:
      "All threat detectors load from ./models with hot-reload support for iterative experimentation.",
    icon: CheckCircle2,
    color: "text-green-400",
  },
  {
    title: "Optional T-Pot",
    description:
      "Integrate a local T-Pot honeypot for attacker telemetry without exposing any AWS services.",
    icon: Network,
    color: "text-amber-300",
  },
];

const steps = [
  "Start the stack with `docker-compose up -d` from the repo root.",
  "Load local models into ./models (see docs/ml/local-models.md).",
  "Add agents or honeypot credentials under Agents → Enrollment.",
  "Validate data flow from the health page and incident feed.",
];

export function QuickStartOnboarding({ onComplete }: { onComplete?: () => void }) {
  const router = useRouter();

  const handleOpenAgents = () => {
    onComplete?.();
    router.push("/agents");
  };

  const handleBackToDashboard = () => {
    onComplete?.();
    router.push("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black py-10 px-6">
      <div className="max-w-5xl mx-auto space-y-10">
        <header className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Shield className="w-12 h-12 text-blue-400" />
            <h1 className="text-4xl font-bold text-white">Local-First Onboarding</h1>
          </div>
          <p className="text-gray-300 max-w-3xl mx-auto">
            Cloud auto-discovery and AWS integrations have been removed. Run the full Mini-XDR stack
            locally, load models from disk, and enroll agents or honeypot connectors directly.
          </p>
        </header>

        <div className="grid md:grid-cols-3 gap-6">
          {highlights.map(({ title, description, icon: Icon, color }) => (
            <div
              key={title}
              className="bg-gray-800/40 border border-gray-700 rounded-xl p-6 shadow-lg shadow-black/40"
            >
              <Icon className={`w-10 h-10 ${color} mb-4`} />
              <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
            </div>
          ))}
        </div>

        <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-8 shadow-xl shadow-black/40">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-6 h-6 text-amber-300" />
            <h2 className="text-2xl font-bold text-white">Get Running in Minutes</h2>
          </div>
          <p className="text-gray-300 mb-6">
            Follow these steps to bring up the local environment and start validating detections:
          </p>
          <ol className="space-y-3">
            {steps.map((step, idx) => (
              <li
                key={step}
                className="flex items-start gap-3 text-gray-200 bg-gray-900/40 border border-gray-800 rounded-lg p-3"
              >
                <span className="text-sm text-blue-300 font-semibold">{idx + 1}</span>
                <span className="text-sm leading-relaxed">{step}</span>
              </li>
            ))}
          </ol>

          <div className="flex flex-col sm:flex-row gap-4 mt-8">
            <ActionButton onClick={handleOpenAgents} icon={<Network className="w-5 h-5" />}>
              Open Agents & Enrollment
            </ActionButton>
            <ActionButton
              type="button"
              variant="secondary"
              onClick={handleBackToDashboard}
              icon={<Server className="w-5 h-5" />}
            >
              Back to Dashboard
            </ActionButton>
          </div>
        </div>
      </div>
    </div>
  );
}
