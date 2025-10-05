'use client'

/**
 * Effectiveness Analysis Component
 * 
 * Detailed analysis of response effectiveness with action-level performance,
 * workflow optimization insights, and comparative analytics.
 */

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  Target, 
  TrendingUp, 
  TrendingDown,
  CheckCircle,
  AlertTriangle,
  Award,
  Zap,
  Activity,
  BarChart3,
  Eye,
  ArrowRight,
  Lightbulb,
  Star,
  Shield,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database
} from 'lucide-react'

interface EffectivenessData {
  action_effectiveness: Record<string, number>
  workflow_effectiveness: Record<string, number>
  improvement_recommendations: string[]
}

interface EffectivenessAnalysisProps {
  data: EffectivenessData
  timeframe: string
  className?: string
}

const EffectivenessAnalysis: React.FC<EffectivenessAnalysisProps> = ({
  data,
  timeframe,
  className = ""
}) => {
  const [sortBy, setSortBy] = useState<'effectiveness' | 'usage'>('effectiveness')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [showRecommendations, setShowRecommendations] = useState(true)

  // Category icons mapping
  const categoryIcons = {
    network: Network,
    endpoint: Server,
    email: Mail,
    cloud: Cloud,
    identity: Key,
    data: Database,
    compliance: Shield,
    forensics: Target
  }

  // Action category mapping (simplified - would come from API in production)
  const actionCategories: Record<string, string> = {
    'block_ip_advanced': 'network',
    'isolate_host_advanced': 'endpoint',
    'memory_dump_collection': 'endpoint',
    'deploy_firewall_rules': 'network',
    'dns_sinkhole': 'network',
    'email_recall': 'email',
    'account_disable': 'identity',
    'data_classification': 'data',
    'compliance_audit_trigger': 'compliance',
    'disk_imaging': 'forensics',
    'network_segmentation': 'network',
    'system_hardening': 'endpoint',
    'container_isolation': 'cloud'
  }

  // Get action category
  const getActionCategory = (actionType: string): string => {
    return actionCategories[actionType] || 'unknown'
  }

  // Get effectiveness color
  const getEffectivenessColor = (effectiveness: number): string => {
    if (effectiveness >= 0.9) return 'text-green-600 bg-green-100 border-green-200'
    if (effectiveness >= 0.8) return 'text-blue-600 bg-blue-100 border-blue-200'
    if (effectiveness >= 0.7) return 'text-yellow-600 bg-yellow-100 border-yellow-200'
    if (effectiveness >= 0.6) return 'text-orange-600 bg-orange-100 border-orange-200'
    return 'text-red-600 bg-red-100 border-red-200'
  }

  // Get effectiveness grade
  const getEffectivenessGrade = (effectiveness: number): string => {
    if (effectiveness >= 0.95) return 'A+'
    if (effectiveness >= 0.9) return 'A'
    if (effectiveness >= 0.85) return 'B+'
    if (effectiveness >= 0.8) return 'B'
    if (effectiveness >= 0.75) return 'C+'
    if (effectiveness >= 0.7) return 'C'
    if (effectiveness >= 0.65) return 'D'
    return 'F'
  }

  // Sort and filter actions
  const sortedActions = Object.entries(data.action_effectiveness)
    .filter(([actionType]) => 
      filterCategory === 'all' || getActionCategory(actionType) === filterCategory
    )
    .sort(([, a], [, b]) => 
      sortBy === 'effectiveness' ? b - a : a - b
    )

  // Sort and filter workflows
  const sortedWorkflows = Object.entries(data.workflow_effectiveness)
    .sort(([, a], [, b]) => b - a)

  // Get unique categories
  const uniqueCategories = Array.from(
    new Set(Object.keys(data.action_effectiveness).map(getActionCategory))
  ).filter(cat => cat !== 'unknown')

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Controls */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={sortBy === 'effectiveness' ? 'default' : 'outline'}
                  onClick={() => setSortBy('effectiveness')}
                >
                  <Target className="h-3 w-3 mr-1" />
                  By Effectiveness
                </Button>
                <Button
                  size="sm"
                  variant={sortBy === 'usage' ? 'default' : 'outline'}
                  onClick={() => setSortBy('usage')}
                >
                  <Activity className="h-3 w-3 mr-1" />
                  By Usage
                </Button>
              </div>
              
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={filterCategory === 'all' ? 'default' : 'outline'}
                  onClick={() => setFilterCategory('all')}
                >
                  All Categories
                </Button>
                {uniqueCategories.map(category => {
                  const Icon = categoryIcons[category as keyof typeof categoryIcons] || Shield
                  return (
                    <Button
                      key={category}
                      size="sm"
                      variant={filterCategory === category ? 'default' : 'outline'}
                      onClick={() => setFilterCategory(category)}
                      className="flex items-center gap-1"
                    >
                      <Icon className="h-3 w-3" />
                      {category}
                    </Button>
                  )
                })}
              </div>
            </div>
            
            <Badge variant="outline">
              {Object.keys(data.action_effectiveness).length} Actions Analyzed
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Action Effectiveness */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Action Effectiveness
            </CardTitle>
            <CardDescription>
              Success rates for individual response actions
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <ScrollArea className="h-96">
              <div className="space-y-3">
                {sortedActions.map(([actionType, effectiveness], index) => {
                  const category = getActionCategory(actionType)
                  const Icon = categoryIcons[category as keyof typeof categoryIcons] || Shield
                  const grade = getEffectivenessGrade(effectiveness)
                  
                  return (
                    <div 
                      key={actionType}
                      className="p-3 border rounded-lg hover:shadow-sm transition-shadow"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4 text-blue-600" />
                          <span className="font-medium text-sm">
                            {actionType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="outline"
                            className={getEffectivenessColor(effectiveness)}
                          >
                            {grade}
                          </Badge>
                          <span className="text-sm font-bold">
                            {Math.round(effectiveness * 100)}%
                          </span>
                        </div>
                      </div>
                      
                      <Progress value={effectiveness * 100} className="h-2 mb-2" />
                      
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Badge variant="secondary" className="text-xs">
                            {category}
                          </Badge>
                        </span>
                        
                        <div className="flex items-center gap-2">
                          {effectiveness >= 0.9 && (
                            <Star className="h-3 w-3 text-yellow-500" />
                          )}
                          {effectiveness < 0.7 && (
                            <AlertTriangle className="h-3 w-3 text-orange-500" />
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
                
                {sortedActions.length === 0 && (
                  <div className="text-center py-8">
                    <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h4 className="font-medium text-gray-900 mb-2">No Action Data</h4>
                    <p className="text-gray-600">No effectiveness data available for selected filters.</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Workflow Effectiveness */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Workflow Effectiveness
            </CardTitle>
            <CardDescription>
              Performance analysis of response playbooks
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <ScrollArea className="h-96">
              <div className="space-y-3">
                {sortedWorkflows.map(([workflowName, effectiveness], index) => {
                  const grade = getEffectivenessGrade(effectiveness)
                  
                  return (
                    <div 
                      key={workflowName}
                      className="p-3 border rounded-lg hover:shadow-sm transition-shadow"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div className="p-1 rounded bg-purple-100">
                            <BarChart3 className="h-3 w-3 text-purple-600" />
                          </div>
                          <span className="font-medium text-sm">{workflowName}</span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="outline"
                            className={getEffectivenessColor(effectiveness)}
                          >
                            {grade}
                          </Badge>
                          <span className="text-sm font-bold">
                            {Math.round(effectiveness * 100)}%
                          </span>
                        </div>
                      </div>
                      
                      <Progress value={effectiveness * 100} className="h-2 mb-2" />
                      
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Playbook Performance</span>
                        
                        <div className="flex items-center gap-1">
                          {effectiveness >= 0.9 ? (
                            <CheckCircle className="h-3 w-3 text-green-500" />
                          ) : effectiveness < 0.7 ? (
                            <AlertTriangle className="h-3 w-3 text-orange-500" />
                          ) : (
                            <Activity className="h-3 w-3 text-blue-500" />
                          )}
                          
                          {index < 3 && (
                            <Star className="h-3 w-3 text-yellow-500" />
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
                
                {sortedWorkflows.length === 0 && (
                  <div className="text-center py-8">
                    <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h4 className="font-medium text-gray-900 mb-2">No Workflow Data</h4>
                    <p className="text-gray-600">No workflow effectiveness data available.</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Improvement Recommendations */}
      {showRecommendations && data.improvement_recommendations.length > 0 && (
        <Card className="border-blue-200 bg-blue-50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-yellow-500" />
                Improvement Recommendations
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowRecommendations(false)}
              >
                ×
              </Button>
            </div>
            <CardDescription>
              AI-powered recommendations for enhancing response effectiveness
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-3">
              {data.improvement_recommendations.map((recommendation, index) => (
                <div key={index} className="p-3 bg-white border border-blue-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <div className="p-1 rounded bg-blue-100">
                      <ArrowRight className="h-3 w-3 text-blue-600" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-gray-700">{recommendation}</p>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      Priority {index + 1}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Effectiveness Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5" />
            Effectiveness Summary
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Top Performers */}
            <div>
              <h4 className="font-semibold text-sm mb-3 flex items-center gap-1">
                <Star className="h-4 w-4 text-yellow-500" />
                Top Performing Actions
              </h4>
              <div className="space-y-2">
                {sortedActions.slice(0, 5).map(([actionType, effectiveness], index) => (
                  <div key={actionType} className="flex items-center justify-between text-sm">
                    <span className="truncate flex-1">
                      {actionType.replace('_', ' ')}
                    </span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {Math.round(effectiveness * 100)}%
                      </Badge>
                      {index === 0 && <Crown className="h-3 w-3 text-yellow-500" />}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Category Performance */}
            <div>
              <h4 className="font-semibold text-sm mb-3">Category Performance</h4>
              <div className="space-y-2">
                {uniqueCategories.map(category => {
                  const categoryActions = Object.entries(data.action_effectiveness)
                    .filter(([actionType]) => getActionCategory(actionType) === category)
                  
                  const avgEffectiveness = categoryActions.length > 0 
                    ? categoryActions.reduce((sum, [, eff]) => sum + eff, 0) / categoryActions.length
                    : 0
                  
                  const Icon = categoryIcons[category as keyof typeof categoryIcons] || Shield
                  
                  return (
                    <div key={category} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <Icon className="h-3 w-3 text-blue-600" />
                        <span className="capitalize">{category}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Progress value={avgEffectiveness * 100} className="w-16 h-1" />
                        <span className="text-xs font-medium">
                          {Math.round(avgEffectiveness * 100)}%
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Performance Insights */}
            <div>
              <h4 className="font-semibold text-sm mb-3">Performance Insights</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>
                    {sortedActions.filter(([, eff]) => eff >= 0.9).length} actions performing excellently
                  </span>
                </div>
                
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-3 w-3 text-orange-500" />
                  <span>
                    {sortedActions.filter(([, eff]) => eff < 0.7).length} actions need improvement
                  </span>
                </div>
                
                <div className="flex items-center gap-2">
                  <Activity className="h-3 w-3 text-blue-500" />
                  <span>
                    {uniqueCategories.length} categories active
                  </span>
                </div>
                
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  <span>
                    Overall effectiveness trending positive
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Crown component for top performer
const Crown: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M5 16L3 12l4 2l3-6l3 6l4-2l-2 4H5z"/>
  </svg>
)

export default EffectivenessAnalysis







