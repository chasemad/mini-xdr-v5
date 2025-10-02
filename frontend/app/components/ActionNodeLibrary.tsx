'use client'

/**
 * Action Node Library Component
 * 
 * Provides a draggable palette of available response actions that can be
 * dropped onto the workflow canvas. Organizes actions by category with
 * visual indicators for safety levels and capabilities.
 */

import React, { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
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
  Filter,
  Clock,
  RotateCcw,
  Zap,
  Target,
  AlertTriangle
} from 'lucide-react'

interface ActionNodeLibraryProps {
  actions: Record<string, any>
  onClose: () => void
}

const ActionNodeLibrary: React.FC<ActionNodeLibraryProps> = ({
  actions,
  onClose
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  // Category icons and colors
  const categoryConfig = {
    network: { icon: Network, color: 'text-blue-600', bgColor: 'bg-blue-50', borderColor: 'border-blue-200' },
    endpoint: { icon: Server, color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' },
    email: { icon: Mail, color: 'text-purple-600', bgColor: 'bg-purple-50', borderColor: 'border-purple-200' },
    cloud: { icon: Cloud, color: 'text-orange-600', bgColor: 'bg-orange-50', borderColor: 'border-orange-200' },
    identity: { icon: Key, color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' },
    data: { icon: Database, color: 'text-gray-600', bgColor: 'bg-gray-50', borderColor: 'border-gray-200' },
    compliance: { icon: Shield, color: 'text-indigo-600', bgColor: 'bg-indigo-50', borderColor: 'border-indigo-200' },
    forensics: { icon: Target, color: 'text-pink-600', bgColor: 'bg-pink-50', borderColor: 'border-pink-200' }
  }

  // Safety level colors
  const safetyColors = {
    low: 'bg-green-100 text-green-800 border-green-200',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    high: 'bg-red-100 text-red-800 border-red-200'
  }

  // Filter actions based on search and category
  const filteredActions = useMemo(() => {
    return Object.entries(actions).filter(([actionType, action]) => {
      const matchesSearch = !searchTerm || 
        actionType.toLowerCase().includes(searchTerm.toLowerCase()) ||
        action.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        action.description.toLowerCase().includes(searchTerm.toLowerCase())
      
      const matchesCategory = selectedCategory === 'all' || action.category === selectedCategory
      
      return matchesSearch && matchesCategory
    })
  }, [actions, searchTerm, selectedCategory])

  // Get unique categories
  const categories = useMemo(() => {
    const cats = new Set(Object.values(actions).map((action: any) => action.category))
    return Array.from(cats)
  }, [actions])

  // Handle drag start
  const onDragStart = (event: React.DragEvent, actionType: string) => {
    event.dataTransfer.setData('application/reactflow', actionType)
    event.dataTransfer.effectAllowed = 'move'
  }

  return (
    <Card className="shadow-lg border-2">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Action Library
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-3 w-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search actions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-7 pr-3 py-2 border rounded-md text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-1">
          <Button
            variant={selectedCategory === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedCategory('all')}
            className="text-xs h-6"
          >
            All
          </Button>
          {categories.map(category => {
            const config = categoryConfig[category as keyof typeof categoryConfig]
            const Icon = config?.icon || Shield
            return (
              <Button
                key={category}
                variant={selectedCategory === category ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory(category)}
                className="text-xs h-6 flex items-center gap-1"
              >
                <Icon className="h-3 w-3" />
                {category}
              </Button>
            )
          })}
        </div>

        {/* Actions List */}
        <ScrollArea className="h-64">
          <div className="space-y-2">
            {filteredActions.map(([actionType, action]) => {
              const config = categoryConfig[action.category as keyof typeof categoryConfig]
              const Icon = config?.icon || Shield

              return (
                <div
                  key={actionType}
                  draggable
                  onDragStart={(e) => onDragStart(e, actionType)}
                  className={`p-2 rounded-lg border cursor-move hover:shadow-md transition-all ${
                    config?.bgColor || 'bg-gray-50'
                  } ${config?.borderColor || 'border-gray-200'} hover:border-opacity-100`}
                >
                  <div className="flex items-start justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <Icon className={`h-4 w-4 ${config?.color || 'text-gray-600'}`} />
                      <span className="font-semibold text-xs">{action.name}</span>
                    </div>
                    <Badge 
                      variant="outline" 
                      className={`text-xs ${safetyColors[action.safety_level as keyof typeof safetyColors]}`}
                    >
                      {action.safety_level}
                    </Badge>
                  </div>
                  
                  <p className="text-xs text-gray-600 mb-2 line-clamp-2">
                    {action.description}
                  </p>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1 text-gray-500">
                      <Clock className="h-3 w-3" />
                      {Math.floor(action.estimated_duration / 60)}m
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {action.rollback_supported && (
                        <div className="flex items-center gap-1 text-green-600">
                          <RotateCcw className="h-3 w-3" />
                          <span>Rollback</span>
                        </div>
                      )}
                      
                      <Badge variant="outline" className="text-xs">
                        {action.category}
                      </Badge>
                    </div>
                  </div>
                  
                  {/* Drag indicator */}
                  <div className="mt-2 text-center">
                    <span className="text-xs text-gray-400 italic">
                      Drag to canvas to add
                    </span>
                  </div>
                </div>
              )
            })}
            
            {filteredActions.length === 0 && (
              <div className="text-center py-8">
                <AlertTriangle className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-gray-600">No actions found</p>
                <p className="text-xs text-gray-500">Try adjusting your search or category filter</p>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Quick Stats */}
        <div className="pt-2 border-t">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-xs font-bold text-blue-600">{Object.keys(actions).length}</div>
              <div className="text-xs text-gray-500">Total</div>
            </div>
            <div>
              <div className="text-xs font-bold text-green-600">{filteredActions.length}</div>
              <div className="text-xs text-gray-500">Filtered</div>
            </div>
            <div>
              <div className="text-xs font-bold text-orange-600">{categories.length}</div>
              <div className="text-xs text-gray-500">Categories</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default ActionNodeLibrary
