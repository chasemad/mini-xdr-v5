# 🧭 SOC Command Center - Navigation Guide

**Date**: October 7, 2025  
**Purpose**: Clear explanation of navigation structure

---

## 📍 Understanding the Page Structure

### **Two Main Entry Points**

#### 1. **Main Dashboard** (`/` or `localhost:3000`)
**Purpose**: Overview tabs and system metrics  
**URL**: `http://localhost:3000/`

**Features**:
- Threat Overview tab (metrics cards)
- Active Incidents tab (incident cards)
- Intelligence tab
- Response Actions tab
- AI Security Analyst chat on right

**Navigation Sidebar**:
- Threat Overview ✅ (local tab)
- Active Incidents ✅ (local tab)
- Threat Intel ✅ (local tab)
- **Threat Hunting** → Links to `/hunt` 🔗
- **Forensics** → Links to `/investigations` 🔗
- Response Actions ✅ (local tab)
- **Workflow Automation** → Links to `/workflows` 🔗
- **3D Visualization** → Links to `/visualizations` 🔗

---

#### 2. **Incidents List** (`/incidents` or `localhost:3000/incidents`)
**Purpose**: List of ALL incidents  
**URL**: `http://localhost:3000/incidents`

**Features**:
- Full list of incidents (cards)
- Click any incident → Enterprise incident detail page
- Same SOC Command sidebar
- System status

**Navigation Sidebar** (UPDATED):
- **Threat Overview** → Links to `/` 🔗
- **Active Incidents** ✅ (current page)
- **Threat Intel** → Links to `/intelligence` 🔗
- **Threat Hunting** → Links to `/hunt` 🔗
- **Forensics** → Links to `/investigations` 🔗
- **Response Actions** → Links to `/` 🔗
- **Workflow Automation** → Links to `/workflows` 🔗
- **3D Visualization** → Links to `/visualizations` 🔗

---

## 🎯 Key Pages Explained

### Page Structure

```
localhost:3000/
├── /                           Main Dashboard (tabs: overview, incidents, intel, response)
├── /incidents                  Incidents List (full list of incidents)
│   └── /incident/14            Enterprise Incident Detail (NEW 2-column UI)
├── /hunt                       Threat Hunting Platform (query builder, analytics)
├── /workflows                  Workflow Automation (NLP chat, designer, templates)
├── /visualizations             3D Threat Globe
├── /intelligence               Threat Intelligence (coming soon)
└── /investigations             Forensics & Investigations (coming soon)
```

---

## 🔀 Navigation Flow

### Flow 1: From Main Dashboard
```
1. Start at: localhost:3000 (Main Dashboard)
2. Click "Threat Hunting" in sidebar
3. Go to: localhost:3000/hunt (Full hunting interface)
```

### Flow 2: From Incident Detail
```
1. Viewing: localhost:3000/incidents/incident/14 (Enterprise UI)
2. Click back button (← arrow)
3. Go to: localhost:3000/incidents (Incidents List)
4. Click "Threat Overview" in sidebar
5. Go to: localhost:3000/ (Main Dashboard)
```

### Flow 3: Access Workflows
```
Option A: From anywhere, click "Workflow Automation" in sidebar
Option B: Go directly to: localhost:3000/workflows

Result: See NLP Chat interface for workflow creation
```

---

## 🗺️ Sidebar Navigation Reference

**All pages now have consistent navigation**:

| Menu Item | Link Destination | Type |
|-----------|-----------------|------|
| **Threat Overview** | `/` | Main dashboard |
| **Active Incidents** | `/incidents` | Incident list |
| **Threat Intel** | `/intelligence` | Dedicated page |
| **Threat Hunting** | `/hunt` | 🎯 **Full hunting platform** |
| **Forensics** | `/investigations` | Dedicated page |
| **Response Actions** | `/` (Response tab) | Main dashboard tab |
| **Workflow Automation** | `/workflows` | 🤖 **NLP chat + designer** |
| **3D Visualization** | `/visualizations` | Interactive globe |

---

## ✅ **What Was Fixed**

### Before (Confusing)
- Incidents page had local tabs for "Threat Overview", "Hunting", etc.
- Clicking those buttons showed placeholder content
- No way to access full features from incidents page
- Navigation inconsistent between pages

### After (Clear)
- ✅ All navigation items are **links to dedicated pages**
- ✅ Clicking "Threat Hunting" always goes to `/hunt`
- ✅ Clicking "Workflow Automation" always goes to `/workflows`
- ✅ Clicking "Threat Overview" goes to main dashboard
- ✅ Consistent navigation across ALL pages
- ✅ No more placeholder tabs

---

## 🎯 Quick Access Guide

### Want to see the NLP Workflow Chat?
```
http://localhost:3000/workflows
```
Then click "Natural Language" tab

### Want to do Threat Hunting?
```
http://localhost:3000/hunt
```
Then click "Interactive Hunt" tab

### Want to see Enterprise Incident UI?
```
http://localhost:3000/incidents
```
Then click any incident

### Want to see 3D Threat Globe?
```
http://localhost:3000/visualizations
```

---

## 📊 Current Page Inventory

### ✅ **Fully Functional Pages**
1. **Main Dashboard** (`/`) - Metrics and overview
2. **Incidents List** (`/incidents`) - All incidents
3. **Enterprise Incident Detail** (`/incidents/incident/[id]`) - ⭐ NEW 2-column UI
4. **Threat Hunting** (`/hunt`) - Full hunt platform
5. **Workflow Automation** (`/workflows`) - NLP chat + designer
6. **3D Visualization** (`/visualizations`) - Interactive globe

### 🚧 **Placeholder/Coming Soon**
1. **Threat Intel** (`/intelligence`) - Basic page exists
2. **Forensics** (`/investigations`) - Basic page exists

---

## ✅ **Verification**

**Test the navigation**:

1. **From incidents list**:
   - Click "Threat Overview" → Should go to `/`
   - Click "Threat Hunting" → Should go to `/hunt`
   - Click "Workflow Automation" → Should go to `/workflows`

2. **From main dashboard**:
   - Click "Active Incidents" → Should go to `/incidents`
   - Click "Threat Hunting" → Should go to `/hunt`
   - Click "Workflow Automation" → Should go to `/workflows`

3. **From hunt page**:
   - Click "Active Incidents" → Should go to `/incidents`
   - Click "Workflow Automation" → Should go to `/workflows`
   - Click "Threat Overview" → Should go to `/`

---

## 🎉 **Now Refresh Your Browser!**

After refreshing, when you click the back button from an incident:
1. You'll be on `/incidents` (incidents list)
2. Sidebar now has **links** (not placeholder tabs)
3. Click "Threat Overview" → Go to main dashboard
4. Click "Threat Hunting" → Go to full hunt interface
5. Click "Workflow Automation" → Go to NLP chat

---

**Status**: ✅ All navigation fixed and consistent  
**All Pages**: ✅ Have proper sidebar navigation  
**No More**: ❌ Confusing placeholder tabs

