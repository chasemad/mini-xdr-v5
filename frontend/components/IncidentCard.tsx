import Link from "next/link";

interface Incident {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
}

interface IncidentCardProps {
  incident: Incident;
}

function StatusBadge({ status, autoContained }: { status: string; autoContained: boolean }) {
  const getStatusColor = () => {
    switch (status) {
      case "open":
        return "bg-yellow-100 text-yellow-800";
      case "contained":
        return "bg-red-100 text-red-800";
      case "dismissed":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor()}`}>
        {status}
      </span>
      {autoContained && (
        <span className="px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
          AUTO
        </span>
      )}
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const getSeverityColor = () => {
    switch (severity) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-orange-100 text-orange-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor()}`}>
      {severity}
    </span>
  );
}

export default function IncidentCard({ incident }: IncidentCardProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <Link href={`/incidents/${incident.id}`}>
      <div className="p-6 rounded-2xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow cursor-pointer">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Incident #{incident.id}
            </h3>
            <p className="text-sm text-gray-600">{formatDate(incident.created_at)}</p>
          </div>
          <StatusBadge status={incident.status} autoContained={incident.auto_contained} />
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">Source IP:</span>
            <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
              {incident.src_ip}
            </span>
          </div>

          <div>
            <span className="text-sm font-medium text-gray-700">Reason:</span>
            <p className="text-sm text-gray-600 mt-1">{incident.reason}</p>
          </div>

          {incident.triage_note && (
            <div className="mt-4 p-3 bg-gray-50 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <SeverityBadge severity={incident.triage_note.severity} />
                <span className="text-xs text-gray-500">
                  Recommend: {incident.triage_note.recommendation}
                </span>
              </div>
              <p className="text-sm text-gray-700">{incident.triage_note.summary}</p>
            </div>
          )}
        </div>
      </div>
    </Link>
  );
}
