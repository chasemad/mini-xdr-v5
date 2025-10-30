# Mini-XDR UI Guidelines

## Overview

Mini-XDR follows a world-class design system inspired by xAI, Tesla, and Apple. This document outlines the complete design system, component usage, and implementation guidelines.

## 🎨 Design System

### Color Palette

**Dark-First Theme (Default)**
```css
/* Surface Hierarchy */
--bg: 222 84% 5%;           /* Deepest background */
--surface-0: 217 32% 12%;   /* Cards, modals, main containers */
--surface-1: 217 32% 16%;   /* Secondary elements, hover states */
--surface-2: 217 32% 22%;   /* Tertiary elements, borders */

/* Typography */
--text: 0 0% 100%;           /* Primary text */
--text-muted: 215 20% 72%;   /* Secondary text */
--text-subtle: 215 20% 58%;  /* Tertiary text */

/* Semantic Colors */
--primary: 217 91% 60%;      /* Actions, links, focus */
--info: 199 89% 58%;         /* Information, notifications */
--success: 142 76% 46%;      /* Positive states */
--warning: 38 92% 56%;       /* Warning states */
--danger: 0 84% 66%;         /* Error states, critical alerts */

/* Severity Scale (for incidents/charts) */
--severity-critical: var(--danger);
--severity-high: 25 95% 60%;
--severity-med: var(--warning);
--severity-low: var(--success);
--severity-info: var(--info);
```

**Light Theme (Optional)**
```css
--bg: 0 0% 100%;
--surface-0: 0 0% 100%;
--surface-1: 217 32% 96%;
--surface-2: 217 32% 92%;
--text: 222 47% 11%;
--text-muted: 217 15% 45%;
--text-subtle: 217 15% 62%;
```

### Typography

**Font Stack**
- **Primary**: Geist Sans (system-optimized)
- **Mono**: Geist Mono (code, data)
- **Fallback**: Inter → system fonts

**Scale (optical sizing)**
```css
--text-display: 2rem (28px);   /* Headlines, hero text */
--text-title: 1.5rem (22px);   /* Section headers */
--text-body: 1rem (14px);      /* Body text */
--text-ui: 0.875rem (12px);    /* Labels, metadata */
--text-caption: 0.75rem (11px); /* Captions, timestamps */
```

**Typography Rules**
- Line heights: 1.4–1.6 for body, 1.2 for headlines
- Letter spacing: -0.01em on display sizes for density
- Weight: Regular (400) default, Medium (500) for emphasis, Semibold (600) for UI labels

### Spacing & Layout

**8pt Base Grid**
```css
--space-1: 0.25rem (4px);
--space-2: 0.5rem (8px);
--space-3: 0.75rem (12px);
--space-4: 1rem (16px);
--space-6: 1.5rem (24px);
--space-8: 2rem (32px);
--space-12: 3rem (48px);
--space-16: 4rem (64px);
```

**Layout Grid**
- Container: max-width 1520px
- Columns: 12-column system
- Gutters: 24–32px between columns
- Content width: 12/12 on mobile, 10/12 on tablet, 8/12 on desktop

### Radii & Shadows

**Border Radius**
```css
--radius-sm: 0.375rem (6px);   /* Buttons, inputs */
--radius-md: 0.5rem (8px);     /* Cards, containers */
--radius-lg: 0.75rem (12px);   /* Large surfaces */
--radius-xl: 1rem (16px);      /* Special elements */
--radius-2xl: 1.5rem (24px);   /* Hero cards, modals */
```

**Shadow System (2-layer max)**
```css
--shadow-sm: 0 1px 3px 0 rgb(15 23 42 / 0.08);
--shadow-md: 0 8px 16px -4px rgb(15 23 42 / 0.12);
--shadow-lg: 0 18px 30px -10px rgb(15 23 42 / 0.18);
--shadow-xl: 0 36px 60px -20px rgb(15 23 42 / 0.25);
--shadow-card: 0 24px 44px -12px rgb(15 23 42 / 0.16);
```

### Motion

**Principles**
- Use motion to explain state changes, not for decoration
- Respect `prefers-reduced-motion`
- Ease-out for entrances (120–180ms)
- Ease-in for exits (80–120ms)

**Keyframes**
```css
@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slide-up {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
```

## 🧩 Component Guidelines

### AppShell (Layout)

**Structure**
```tsx
<AppShell breadcrumbs={[{ label: "Incidents", href: "/incidents" }]}>
  {/* Page content */}
</AppShell>
```

**Features**
- Collapsible sidebar with tooltips
- Persistent Copilot dock (right side)
- Breadcrumb navigation
- Command palette (⌘K)
- Responsive: sidebar → hamburger on mobile

### Data Table (TanStack)

**Configuration**
```tsx
<DataTable
  columns={columns}
  data={data}
  enableSorting
  enableFiltering
  enablePagination
  enableRowSelection
  enableVirtualization
  searchPlaceholder="Search incidents..."
  onRowClick={(row) => router.push(`/incidents/${row.id}`)}
/>
```

**Features**
- Column sorting, filtering, resizing
- Row selection with checkboxes
- Global search
- Virtualized for performance
- Responsive pagination

### KPI Tiles

**Usage**
```tsx
<KpiTile
  title="Total Incidents"
  value={metrics.total}
  delta={+12.5}
  deltaLabel="from yesterday"
  icon={AlertTriangle}
  trend="up"
/>
```

**Variants**
- `KpiTile`: Basic metric display
- `SecurityKpiTile`: Severity-aware styling

### Status & Severity

**Status Badges**
```tsx
<StatusChip status="success" label="Active" />
<SeverityBadge severity="critical" />
```

**Color Mapping**
- Success: `bg-success/15 text-success border-success/40`
- Warning: `bg-warning/15 text-warning border-warning/40`
- Error: `bg-danger/15 text-danger border-danger/40`
- Info: `bg-info/15 text-info border-info/40`

### Buttons & Actions

**Primary Actions**
```tsx
<Button variant="default">Primary Action</Button>
<ActionButton variant="primary" icon={<Zap />}>
  Execute
</ActionButton>
```

**Hierarchy**
- Primary: High-impact actions
- Secondary: Alternative actions
- Ghost: Low-priority, informational
- Danger: Destructive actions

### Forms & Inputs

**Input Styling**
```tsx
<Input
  className="bg-surface-1 border-border focus:border-primary"
  placeholder="Enter value..."
/>
```

**Validation States**
- Default: `border-border`
- Focus: `border-primary ring-2 ring-primary/20`
- Error: `border-danger`
- Success: `border-success`

## 📱 Responsive Design

### Breakpoints
```css
--screen-sm: 640px;   /* Mobile → Tablet */
--screen-md: 768px;   /* Tablet → Desktop */
--screen-lg: 1024px;  /* Desktop → Large */
--screen-xl: 1280px;  /* Large → XL */
--screen-2xl: 1536px; /* XL → 2XL */
```

### Layout Patterns

**Mobile First**
- Single column on mobile
- Stack vertically, expand horizontally
- Touch targets: minimum 44px
- Swipe gestures where appropriate

**Tablet (768px+)**
- 2-column layouts
- Sidebar collapses with hamburger
- Touch-optimized spacing

**Desktop (1024px+)**
- Multi-column layouts
- Full sidebar navigation
- Hover states and tooltips
- Keyboard shortcuts active

## ♿ Accessibility

### WCAG 2.2 AA Compliance

**Color Contrast**
- Normal text: 4.5:1 minimum
- Large text: 3:1 minimum
- UI components: 3:1 minimum

**Focus Management**
```css
.focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}
```

**Keyboard Navigation**
- Tab order follows visual order
- Enter/Space activate buttons
- Escape closes modals/drawers
- Arrow keys navigate lists/menus

**Screen Reader Support**
- Semantic HTML elements
- ARIA labels where needed
- Live regions for dynamic content
- Skip links for navigation

## 🎯 Interaction Patterns

### Loading States
```tsx
// Skeleton loading
<div className="animate-pulse bg-surface-2 rounded h-4 w-24" />

// Spinner with context
<div className="flex items-center gap-2">
  <Loader2 className="animate-spin" />
  <span>Analyzing threat...</span>
</div>
```

### Empty States
```tsx
<div className="text-center py-12">
  <FileX className="mx-auto h-12 w-12 text-text-muted" />
  <h3 className="text-text font-medium">No incidents found</h3>
  <p className="text-text-muted">Try adjusting your filters</p>
</div>
```

### Error States
```tsx
<Alert className="border-danger/20 bg-danger/5">
  <AlertTriangle className="text-danger" />
  <AlertDescription className="text-danger">
    Failed to load incidents. <button className="underline">Retry</button>
  </AlertDescription>
</Alert>
```

## 🔧 Implementation Guidelines

### CSS Architecture

**Token Usage**
```tsx
// ✅ Do: Use semantic tokens
<div className="bg-surface-0 text-text border border-border">

// ❌ Don't: Hardcode colors
<div className="bg-gray-900 text-white border border-gray-700">
```

**Class Organization**
```tsx
// ✅ Group related classes
<div className="rounded-2xl border border-border bg-surface-0 p-6 shadow-card">

// ❌ Don't: Scatter classes
<div className="rounded-2xl border border-border bg-surface-0 p-6 shadow-card hover:shadow-lg">
```

### Component Patterns

**Composition Over Props**
```tsx
// ✅ Prefer composition
<Card>
  <CardHeader>
    <CardTitle>Incident Details</CardTitle>
  </CardHeader>
  <CardContent>{children}</CardContent>
</Card>

// ❌ Avoid prop explosion
<Card title="Incident Details" showHeader variant="outlined" size="large">
```

**Consistent Naming**
```tsx
// Component files: PascalCase
// Utilities: camelCase
// Hooks: useCamelCase
// Constants: SCREAMING_SNAKE
```

### Performance

**Bundle Splitting**
```tsx
// Lazy load heavy components
const IncidentDetail = lazy(() => import('./IncidentDetail'))
```

**Image Optimization**
```tsx
// Next.js Image component
<Image src={src} width={400} height={300} alt="Description" />
```

**Animation Performance**
```css
/* Use transform and opacity for smooth animations */
.element {
  transition: transform 0.2s ease-out, opacity 0.2s ease-out;
}
```

## 📊 Data Visualization

### Charts (Recharts)

**Color Usage**
```tsx
const severityColors = [
  'var(--severity-info)',
  'var(--severity-low)',
  'var(--severity-med)',
  'var(--severity-high)',
  'var(--severity-critical)'
];
```

**Accessibility**
- Include `<title>` and `<desc>` elements
- Use semantic colors for data encoding
- Provide text alternatives for complex visualizations

## 🚀 Development Workflow

### Component Checklist

**Before PR:**
- [ ] Uses semantic tokens, not hardcoded colors
- [ ] Responsive across breakpoints
- [ ] Keyboard accessible
- [ ] Loading and error states handled
- [ ] Performance optimized (no unnecessary re-renders)
- [ ] Follows naming conventions

### Testing Requirements

**Unit Tests**
```tsx
describe('KpiTile', () => {
  it('displays value and delta correctly', () => {
    render(<KpiTile value={100} delta={10} />);
    expect(screen.getByText('100')).toBeInTheDocument();
    expect(screen.getByText('+10%')).toBeInTheDocument();
  });
});
```

**Visual Regression**
```ts
// Playwright visual tests
test('dashboard looks correct', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot('dashboard.png');
});
```

**Accessibility Testing**
```ts
// Axe integration
test('dashboard is accessible', async ({ page }) => {
  await page.goto('/');
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toHaveLength(0);
});
```

## 📚 Resources

### Design Tools
- **Figma**: Component library and design system
- **Storybook**: Interactive component documentation
- **Chromatic**: Visual regression testing

### Code Quality
- **ESLint**: Code linting with custom rules
- **Prettier**: Code formatting
- **TypeScript**: Type safety
- **Husky**: Git hooks for quality gates

### Documentation
- **Component Documentation**: Storybook stories
- **API Documentation**: OpenAPI specs
- **Design System**: This document
- **Architecture**: ADR system

---

## 📞 Getting Help

**Design Questions**
- Post in #design-system Slack channel
- Reference this document first

**Implementation Questions**
- Check existing components in Storybook
- Review similar implementations
- Ask in #frontend Slack channel

**Breaking Changes**
- Create ADR for significant changes
- Update this document
- Notify team of breaking changes
