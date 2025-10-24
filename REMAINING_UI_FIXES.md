# Remaining UI/UX Fixes

## Completed ✅
1. **Incidents Page** - Fixed missing Globe import
2. **Agents Page** - Wrapped with DashboardLayout
3. **Analytics Page** - Wrapped with DashboardLayout

## Remaining Work

### 4. Workflows Page (Most Complex)
**File:** `/app/workflows/page.tsx`

**Current Issues:**
- Lines 338-456: Has complete custom sidebar with "SOC Command" branding
- Different navigation items than DashboardLayout
- Custom AI agents status panel
- Custom workflow status section

**Required Changes:**
1. Add import: `import { DashboardLayout } from "@/components/DashboardLayout"`
2. Remove entire sidebar section (lines 340-456)
3. Wrap return statement with: `<DashboardLayout breadcrumbs={[{ label: "Workflows" }]}>`
4. Keep all main content (lines 458-809)
5. Close with `</DashboardLayout>`

### 5. Visualizations Page
**File:** `/app/visualizations/page.tsx`

**Current Structure:**
- Very large file (~600+ lines)
- Has custom sidebar implementation
- Complex 3D visualization dashboard

**Required Changes:**
1. Add import: `import { DashboardLayout } from "@/components/DashboardLayout"`
2. Remove custom sidebar (if present)
3. Wrap main content with DashboardLayout
4. Use breadcrumb: `breadcrumbs={[{ label: "3D Visualizations" }]}`

### 6. Automations Page
**File:** `/app/automations/page.tsx`

**Current Structure:**
- Custom layout starting around line 52

**Required Changes:**
1. Add import: `import { DashboardLayout } from "@/components/DashboardLayout"`
2. Replace outer `<div>` wrapper with DashboardLayout
3. Remove any custom headers/titles (breadcrumbs will handle it)
4. Use breadcrumb: `breadcrumbs={[{ label: "Automations" }]}`

## Testing Checklist

After completing all fixes:

### Local Testing
- [ ] Run `npm run dev`
- [ ] Visit each page and verify:
  - [ ] Incidents (/incidents) - Loads properly, no blank screen
  - [ ] Agents (/agents) - Shows unified navigation
  - [ ] Analytics (/analytics) - Shows unified navigation
  - [ ] Workflows (/workflows) - Shows unified navigation (no "SOC Command")
  - [ ] Visualizations (/visualizations) - Shows unified navigation
  - [ ] Automations (/automations) - Shows unified navigation
- [ ] Test user dropdown on each page:
  - [ ] Click avatar in top-right
  - [ ] Verify "Settings" link appears
  - [ ] Verify "Sign Out" button appears
  - [ ] Test click-outside to close
- [ ] Verify all page functionality still works

### AWS Deployment
1. Commit changes:
   ```bash
   git add .
   git commit -m "fix(ui): unify all pages with DashboardLayout navigation"
   git push origin main
   ```

2. SSH to EC2 build instance:
   ```bash
   ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-IP>
   cd /home/ec2-user/mini-xdr-v2
   git pull origin main
   ```

3. Build and push Docker image:
   ```bash
   cd frontend
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 116912495274.dkr.ecr.us-east-1.amazonaws.com

   docker build \
     --build-arg NEXT_PUBLIC_API_BASE="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
     --build-arg NEXT_PUBLIC_API_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
     --build-arg VERSION="1.1.2" \
     -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.2 \
     -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest \
     .

   docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.2
   docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest
   ```

4. Update Kubernetes:
   ```bash
   kubectl set image deployment/mini-xdr-frontend \
     frontend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.2 \
     -n mini-xdr

   kubectl rollout status deployment/mini-xdr-frontend -n mini-xdr
   ```

5. Verify deployment:
   ```bash
   kubectl get pods -n mini-xdr
   kubectl logs -f deployment/mini-xdr-frontend -n mini-xdr --tail=50
   ```

6. Test in browser (force refresh with Cmd+Shift+R or Ctrl+Shift+R)

## Pattern for Wrapping Pages

**Before:**
```tsx
export default function PageName() {
  return (
    <div className="some-custom-class">
      <h1>Page Title</h1>
      {/* page content */}
    </div>
  );
}
```

**After:**
```tsx
import { DashboardLayout } from "@/components/DashboardLayout";

export default function PageName() {
  return (
    <DashboardLayout breadcrumbs={[{ label: "Page Name" }]}>
      {/* page content - remove custom title, keep functionality */}
    </DashboardLayout>
  );
}
```

## Expected Final Result

All pages should have:
- ✅ Consistent left sidebar with Mini-XDR branding
- ✅ Same navigation menu items
- ✅ User dropdown in top-right with Settings & Sign Out
- ✅ Breadcrumb navigation showing current page
- ✅ No custom sidebars or navigation
- ✅ No "SOC Command" or other alternate branding
