import React from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import MLMonitoringDashboard from './ml-monitoring'
import ExplainableAIDashboard from './explainable-ai'
import { Brain, BarChart3 } from 'lucide-react'
import { DashboardLayout } from '@/components/DashboardLayout'

export default function AnalyticsPage() {
  return (
    <DashboardLayout breadcrumbs={[{ label: "Analytics" }]}>
      <div className="space-y-6">
        <div>
          <p className="text-lg text-gray-400">
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
    </DashboardLayout>
  )
}
