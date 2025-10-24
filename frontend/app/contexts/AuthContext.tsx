'use client'

/**
 * Authentication Context
 * Manages user authentication state and organization information
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { login as apiLogin, getCurrentUser, logout as apiLogout } from '../lib/api'

// Types
interface User {
  id: number
  email: string
  full_name: string
  role: string
  is_active: boolean
  organization_id: number
}

interface Organization {
  id: number
  name: string
  slug: string
  status: string
}

interface AuthContextType {
  user: User | null
  organization: Organization | null
  isAuthenticated: boolean
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
}

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Provider component
interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [organization, setOrganization] = useState<Organization | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  // Fetch user info on mount
  useEffect(() => {
    const token = localStorage.getItem('access_token')
    if (token) {
      fetchUserInfo()
    } else {
      setLoading(false)
    }
  }, [])

  const fetchUserInfo = async () => {
    try {
      const data = await getCurrentUser()
      setUser(data.user)
      setOrganization(data.organization)
      console.log('✅ User authenticated:', data.user.email, 'Org:', data.organization.name)
    } catch (error) {
      console.error('❌ Failed to fetch user info:', error)
      // Clear invalid tokens
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      setUser(null)
      setOrganization(null)
    } finally {
      setLoading(false)
    }
  }

  const login = async (email: string, password: string) => {
    try {
      const data = await apiLogin(email, password)
      
      // Store tokens
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      
      // Fetch user info
      await fetchUserInfo()
      
      console.log('✅ Login successful')
      router.push('/')
    } catch (error) {
      console.error('❌ Login failed:', error)
      throw error
    }
  }

  const logout = async () => {
    try {
      await apiLogout()
    } catch (error) {
      console.error('❌ Logout API error (continuing anyway):', error)
    }
    
    // Clear state
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setUser(null)
    setOrganization(null)
    
    console.log('✅ Logged out')
    router.push('/login')
  }

  const refreshUser = async () => {
    await fetchUserInfo()
  }

  const value: AuthContextType = {
    user,
    organization,
    isAuthenticated: !!user,
    loading,
    login,
    logout,
    refreshUser
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

// Hook to use the auth context
export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}


