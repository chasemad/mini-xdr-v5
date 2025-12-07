'use client'

/**
 * Action Node Library Component
 *
 * Provides a categorized catalog of available workflow nodes (Triggers, Agents, Apps, Core).
 * Features a sidebar navigation and a grid layout for actions, similar to n8n.
 */

import React, { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import {
  Shield,
  Network,
  Server,
  Mail,
  Cloud,
  Key,
  Database,
  X,
  Search,
  Zap,
  Target,
  Bot,
  Webhook,
  Clock,
  AlertCircle,
  Globe,
  Lock,
  FileSearch,
  Terminal,
  Cpu,
  Layers,
  Box
} from 'lucide-react'
import { WorkflowAction, ActionCategory } from './workflow/workflow-actions'

interface ActionNodeLibraryProps {
  actions: Record<string, WorkflowAction>
  onClose?: () => void
  isEmbedded?: boolean
}

const ActionNodeLibrary: React.FC<ActionNodeLibraryProps> = ({
  actions,
  onClose,
  isEmbedded = false
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [activeTab, setActiveTab] = useState<ActionCategory | 'all'>('all')

  // Navigation Categories
  const navItems = [
    { id: 'all', label: 'All Nodes', icon: Layers },
    { id: 'trigger', label: 'Triggers', icon: Zap },
    { id: 'agent', label: 'AI Agents', icon: Bot },
    { id: 'app', label: 'Integrations', icon: Box },
    { id: 'core', label: 'Core Logic', icon: Cpu },
  ]

  // Filter actions based on search and category
  const filteredActions = useMemo(() => {
    return Object.entries(actions).filter(([_, action]) => {
      const matchesSearch = !searchTerm ||
        action.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        action.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (action.subcategory && action.subcategory.toLowerCase().includes(searchTerm.toLowerCase()))

      const matchesCategory = activeTab === 'all' || action.category === activeTab

      return matchesSearch && matchesCategory
    })
  }, [actions, searchTerm, activeTab])

  // Group by Subcategory (Vendor) for Apps
  const groupedActions = useMemo(() => {
    const groups: Record<string, typeof filteredActions> = {}

    filteredActions.forEach(([key, action]) => {
      const groupName = action.subcategory || 'General'
      if (!groups[groupName]) groups[groupName] = []
      groups[groupName].push([key, action])
    })

    return groups
  }, [filteredActions])

  // Handle drag start
  const onDragStart = (event: React.DragEvent, actionType: string) => {
    event.dataTransfer.setData('application/reactflow', actionType)
    event.dataTransfer.effectAllowed = 'move'
  }

  const Content = (
    <div className="flex h-full">
      {/* Left Sidebar Navigation */}
      <div className="w-12 flex flex-col items-center py-4 gap-2 border-r border-white/5 bg-[#0A0A0A]">
        {navItems.map(item => (
          <Button
            key={item.id}
            variant="ghost"
            size="icon"
            onClick={() => setActiveTab(item.id as any)}
            className={`
              w-8 h-8 rounded-lg transition-all
              ${activeTab === item.id
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-slate-500 hover:text-slate-200 hover:bg-white/5'}
            `}
            title={item.label}
          >
            <item.icon size={18} />
          </Button>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0A0A0A]">
        {/* Search Header */}
        <div className="p-3 border-b border-white/5">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
            <Input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 h-9 bg-[#111] border-white/10 text-xs text-slate-200 focus:border-cyan-500/50"
            />
          </div>
        </div>

        {/* Nodes Grid */}
        <ScrollArea className="flex-1 p-3">
          <div className="space-y-6 pb-4">
            {Object.entries(groupedActions).map(([group, items]) => (
              <div key={group}>
                <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 pl-1">
                  {group}
                </h3>
                <div className="grid grid-cols-1 gap-2">
                  {items.map(([actionType, action]) => {
                    const Icon = action.icon || Box

                    return (
                      <div
                        key={actionType}
                        draggable
                        onDragStart={(e) => onDragStart(e, actionType)}
                        className={`
                          group flex items-center gap-3 p-2 rounded-lg border cursor-grab active:cursor-grabbing transition-all
                          bg-[#111] border-white/5 hover:border-cyan-500/30 hover:bg-[#161616]
                        `}
                      >
                        <div className={`
                          p-2 rounded-md bg-[#0A0A0A] border border-white/5 text-slate-400 group-hover:text-cyan-400 group-hover:border-cyan-500/20 transition-colors
                        `}>
                          <Icon size={16} />
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-slate-200 truncate pr-2">
                              {action.name}
                            </span>
                            {action.safety_level === 'high' && (
                              <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 border-red-500/30 text-red-400 bg-red-950/10">
                                High Risk
                              </Badge>
                            )}
                          </div>
                          <p className="text-[10px] text-slate-500 truncate">
                            {action.description}
                          </p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}

            {Object.keys(groupedActions).length === 0 && (
              <div className="text-center py-8">
                <p className="text-xs text-slate-500">No nodes found</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )

  if (isEmbedded) {
    return Content
  }

  return (
    <Card className="shadow-2xl border border-white/10 bg-[#0A0A0A]/95 backdrop-blur-xl text-slate-200 w-[400px] h-[600px] overflow-hidden flex flex-col">
      <CardHeader className="py-3 px-4 border-b border-white/5 shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2 font-semibold text-white">
            <Layers className="h-4 w-4 text-cyan-400" />
            Node Library
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-6 w-6 p-0 hover:bg-white/10 text-slate-400 hover:text-white">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden">
        {Content}
      </CardContent>
    </Card>
  )
}

export default ActionNodeLibrary
