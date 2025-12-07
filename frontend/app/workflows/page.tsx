'use client'

/**
 * Enterprise Workflow Orchestration Platform - "The Neural Grid"
 *
 * Comprehensive workflow management with AI-powered natural language interface,
 * visual drag-and-drop designer, and real-time execution monitoring.
 */

import React from 'react'
import WorkflowCanvas from '../components/WorkflowCanvas'
import { DashboardLayout } from '@/components/DashboardLayout'

const WorkflowsPage: React.FC = () => {
  return (
    <DashboardLayout breadcrumbs={[{ label: "Workflows" }]}>
      {/*
        The Neural Grid Interface
        We remove the standard padding/container to allow the canvas to be full-screen/immersive
      */}
      <div className="absolute inset-0 top-14 z-0">
        <WorkflowCanvas />
      </div>
    </DashboardLayout>
  )
}

export default WorkflowsPage
