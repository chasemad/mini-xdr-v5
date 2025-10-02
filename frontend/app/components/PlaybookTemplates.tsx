'use client'

/**
 * Playbook Templates Component
 * Pre-built response workflow templates for common security scenarios
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Shield,
  Lock,
  Mail,
  Database,
  Cloud,
  AlertTriangle,
  FileText,
  Zap,
  CheckCircle,
  Clock
} from 'lucide-react'

interface PlaybookTemplate {
  id: string
  name: string
  description: string
  category: string
  priority: 'low' | 'medium' | 'high' | 'critical'
  estimatedTime: string
  actions: number
  icon: React.ComponentType<{className?: string}>
  steps: string[]
}

const templates: PlaybookTemplate[] = [
  {
    id: 'emergency-containment',
    name: 'Emergency Containment',
    description: 'Immediate response to active threats with rapid containment and isolation',
    category: 'Incident Response',
    priority: 'critical',
    estimatedTime: '2-5 minutes',
    actions: 4,
    icon: AlertTriangle,
    steps: ['Block attacker IP', 'Isolate affected hosts', 'Alert security team', 'Create incident case']
  },
  {
    id: 'ransomware-response',
    name: 'Ransomware Response',
    description: 'Comprehensive ransomware incident response with data protection',
    category: 'Malware',
    priority: 'critical',
    estimatedTime: '10-15 minutes',
    actions: 7,
    icon: Lock,
    steps: [
      'Isolate all affected systems',
      'Block C2 communications',
      'Backup critical data',
      'Reset compromised credentials',
      'Deploy endpoint protection',
      'Scan entire environment',
      'Document incident timeline'
    ]
  },
  {
    id: 'data-breach-response',
    name: 'Data Breach Response',
    description: 'Rapid response to data exfiltration with access control and forensics',
    category: 'Data Protection',
    priority: 'high',
    estimatedTime: '15-20 minutes',
    actions: 6,
    icon: Database,
    steps: [
      'Encrypt sensitive data',
      'Revoke unauthorized access',
      'Check database integrity',
      'Enable DLP policies',
      'Collect forensic evidence',
      'Alert compliance team'
    ]
  },
  {
    id: 'phishing-investigation',
    name: 'Phishing Investigation',
    description: 'Email threat response with user protection and training',
    category: 'Email Security',
    priority: 'medium',
    actions: 5,
    icon: Mail,
    steps: [
      'Quarantine suspicious emails',
      'Block sender domains',
      'Reset affected user passwords',
      'Provide security training',
      'Create forensic case'
    ]
  },
  {
    id: 'insider-threat-investigation',
    name: 'Insider Threat Investigation',
    description: 'Comprehensive investigation of potential insider threats',
    category: 'Insider Threat',
    priority: 'high',
    estimatedTime: '30-45 minutes',
    actions: 6,
    icon: FileText,
    steps: [
      'Disable user access immediately',
      'Capture forensic evidence',
      'Review access logs',
      'Preserve data for investigation',
      'Document all findings',
      'Alert HR/Legal'
    ]
  },
  {
    id: 'ddos-mitigation',
    name: 'DDoS Mitigation',
    description: 'Rapid response to distributed denial of service attacks',
    category: 'Network Defense',
    priority: 'high',
    estimatedTime: '5-10 minutes',
    actions: 5,
    icon: Cloud,
    steps: [
      'Deploy rate limiting',
      'Block attacking IP ranges',
      'Enable cloud WAF',
      'Scale infrastructure',
      'Monitor traffic patterns'
    ]
  },
  {
    id: 'malware-containment',
    name: 'Malware Containment',
    description: 'Quick containment and removal of malware infections',
    category: 'Malware',
    priority: 'high',
    estimatedTime: '15-20 minutes',
    actions: 6,
    icon: Shield,
    steps: [
      'Isolate infected endpoints',
      'Terminate malicious processes',
      'Collect memory dumps',
      'Scan entire environment',
      'Deploy updated signatures',
      'Update detection rules'
    ]
  },
  {
    id: 'bec-response',
    name: 'BEC Response',
    description: 'Business email compromise investigation and remediation',
    category: 'Email Security',
    priority: 'critical',
    estimatedTime: '20-30 minutes',
    actions: 5,
    icon: Zap,
    steps: [
      'Verify email authenticity',
      'Reset compromised accounts',
      'Review financial transactions',
      'Alert finance team',
      'Implement additional controls'
    ]
  }
]

const priorityColors = {
  low: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
  medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  high: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
  critical: 'bg-red-500/10 text-red-400 border-red-500/30'
}

const priorityLabels = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  critical: 'Critical'
}

interface PlaybookTemplatesProps {
  onSelectTemplate?: (template: PlaybookTemplate) => void
  selectedIncidentId?: number
}

export default function PlaybookTemplates({ onSelectTemplate, selectedIncidentId }: PlaybookTemplatesProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')

  const categories = ['all', ...new Set(templates.map(t => t.category))]

  const filteredTemplates = selectedCategory === 'all'
    ? templates
    : templates.filter(t => t.category === selectedCategory)

  const handleUseTemplate = (template: PlaybookTemplate) => {
    if (onSelectTemplate) {
      onSelectTemplate(template)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Playbook Templates</h2>
          <p className="text-gray-400 mt-1">Pre-built response workflows for common security scenarios</p>
        </div>
        <Badge variant="outline" className="border-blue-500/30 text-blue-400">
          {templates.length} Templates Available
        </Badge>
      </div>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2">
        {categories.map((category) => (
          <Button
            key={category}
            size="sm"
            variant={selectedCategory === category ? "default" : "outline"}
            onClick={() => setSelectedCategory(category)}
            className={selectedCategory === category
              ? "bg-blue-600 hover:bg-blue-700 text-white"
              : "border-gray-700 hover:bg-gray-800 text-gray-300"
            }
          >
            {category === 'all' ? 'All Templates' : category}
          </Button>
        ))}
      </div>

      {/* Templates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates.map((template) => {
          const Icon = template.icon
          return (
            <Card key={template.id} className="bg-gray-900/50 border-gray-700 hover:border-blue-500/50 transition-all hover:shadow-lg hover:shadow-blue-500/10">
              <CardHeader>
                <div className="flex items-start justify-between mb-3">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <Icon className="w-6 h-6 text-blue-400" />
                  </div>
                  <Badge className={`${priorityColors[template.priority]} border px-2 py-0.5`}>
                    {priorityLabels[template.priority]}
                  </Badge>
                </div>
                <CardTitle className="text-white text-lg">{template.name}</CardTitle>
                <CardDescription className="text-gray-400 text-sm">
                  {template.description}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Metadata */}
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-1 text-gray-400">
                    <Zap className="w-4 h-4" />
                    <span>{template.actions} actions</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-400">
                    <Clock className="w-4 h-4" />
                    <span>{template.estimatedTime}</span>
                  </div>
                </div>

                {/* Steps Preview */}
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Workflow Steps:</p>
                  <div className="space-y-1">
                    {template.steps.slice(0, 3).map((step, index) => (
                      <div key={index} className="flex items-start gap-2 text-xs text-gray-400">
                        <CheckCircle className="w-3 h-3 text-green-400 mt-0.5 flex-shrink-0" />
                        <span>{step}</span>
                      </div>
                    ))}
                    {template.steps.length > 3 && (
                      <p className="text-xs text-gray-500 pl-5">
                        +{template.steps.length - 3} more steps...
                      </p>
                    )}
                  </div>
                </div>

                {/* Action Button */}
                <Button
                  onClick={() => handleUseTemplate(template)}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                  disabled={!selectedIncidentId}
                >
                  Use Template
                </Button>

                {!selectedIncidentId && (
                  <p className="text-xs text-center text-gray-500">
                    Select an incident first
                  </p>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}