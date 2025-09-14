import React from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import MLMonitoringDashboard from './ml-monitoring'
import ExplainableAIDashboard from './explainable-ai'
import { Brain, BarChart3, Target } from 'lucide-react'

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-6">
        <div className="mb-6">
          <h1 className="text-4xl font-bold mb-2">Advanced Analytics</h1>
          <p className="text-lg text-muted-foreground">
            Comprehensive monitoring and analysis of ML models, explainable AI insights, and system performance
          </p>
        </div>

        <Tabs defaultValue="ml-monitoring" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="ml-monitoring" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              ML Monitoring
            </TabsTrigger>
            <TabsTrigger value="explainable-ai" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Explainable AI
            </TabsTrigger>
          </TabsList>

          <TabsContent value="ml-monitoring">
            <MLMonitoringDashboard />
          </TabsContent>

          <TabsContent value="explainable-ai">
            <ExplainableAIDashboard />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}