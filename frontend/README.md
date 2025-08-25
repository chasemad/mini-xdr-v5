# Mini-XDR Frontend

Modern Next.js frontend for the Mini-XDR security incident management system.

## Features

- **SOC-Style Interface**: Professional security operations center UI
- **Real-time Updates**: Live incident status and system health
- **Incident Management**: Full lifecycle incident handling
- **Triage Integration**: AI-powered analysis display
- **Response Controls**: Manual and scheduled containment actions
- **Modern Design**: Tailwind CSS with responsive layout

## Setup

### Install Dependencies

```bash
npm install
```

### Configure Environment

```bash
cp env.local .env.local
```

Edit `.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://10.0.0.123:8000
NEXT_PUBLIC_API_KEY=optional_api_key
```

### Run Development Server

```bash
npm run dev
```

Visit http://localhost:3000

## Pages

### Overview (`/`)

System dashboard showing:
- System health status
- Auto-contain toggle
- Total incident count
- Environment status
- Key metrics

### Incidents (`/incidents`)

Incident list page featuring:
- Reverse chronological incident list
- Status and severity indicators
- Triage summary cards
- Quick access to details

### Incident Detail (`/incidents/[id]`)

Comprehensive incident view with:
- Triage analysis and recommendations
- Action controls (contain, unblock, schedule)
- Recent events timeline
- Action history log
- Real-time status updates

## Components

### IncidentCard

Reusable incident list item with:
- Status badges (open/contained/dismissed)
- Auto-contain indicators
- Severity levels (high/medium/low)
- Triage summaries
- Clickable navigation

### StatusBadge

Dynamic status indicators:
- Color-coded by status
- Auto-contain markers
- Severity levels
- Recommendation tags

## API Integration

### API Client (`lib/api.ts`)

Centralized API communication with:
- Environment-based configuration
- Automatic API key handling
- Error handling and retries
- TypeScript interfaces

### Available Methods

```typescript
// Incident management
getIncidents(): Promise<Incident[]>
getIncident(id: number): Promise<IncidentDetail>
containIncident(id: number): Promise<ActionResult>
unblockIncident(id: number): Promise<ActionResult>
scheduleUnblock(id: number, minutes: number): Promise<ScheduleResult>

// System settings
getAutoContainSetting(): Promise<{enabled: boolean}>
setAutoContainSetting(enabled: boolean): Promise<{enabled: boolean}>

// Health monitoring
getHealth(): Promise<HealthStatus>
```

## Styling

### Tailwind CSS

Modern utility-first styling with:
- Consistent color palette
- Responsive breakpoints
- Component variants
- Custom animations

### Design System

- **Cards**: Rounded corners (2xl), subtle shadows
- **Buttons**: Consistent padding, hover states
- **Badges**: Color-coded status indicators
- **Layout**: Grid-based responsive design
- **Typography**: Clear hierarchy, readable fonts

## State Management

### React Hooks

Client-side state using:
- `useState` for component state
- `useEffect` for data fetching
- `useRouter` for navigation
- Custom hooks for API calls

### Data Flow

1. Page components fetch initial data
2. User interactions trigger API calls
3. State updates reflect changes
4. UI re-renders with new data

## Performance

### Optimization Features

- Static generation where possible
- Image optimization
- Code splitting
- Lazy loading
- Caching strategies

### Bundle Analysis

```bash
npm run build
npm run start
```

## Testing

### Development Testing

```bash
# Start development server
npm run dev

# Test API connectivity
curl http://localhost:3000/api/health

# Verify environment variables
npm run build
```

### Manual Testing Checklist

1. **Overview Page**
   - [ ] Health status displays correctly
   - [ ] Auto-contain toggle works
   - [ ] Incident count updates
   - [ ] Environment indicators show

2. **Incidents Page**
   - [ ] Incidents load and display
   - [ ] Status badges show correctly
   - [ ] Triage cards appear
   - [ ] Navigation works

3. **Incident Detail**
   - [ ] Full incident data loads
   - [ ] Triage analysis displays
   - [ ] Action buttons work
   - [ ] Events/actions show

4. **Responsive Design**
   - [ ] Mobile layout works
   - [ ] Tablet breakpoints
   - [ ] Desktop optimization

## Deployment

### Production Build

```bash
npm run build
npm run start
```

### Environment Variables

Production environment requires:

```bash
NEXT_PUBLIC_API_BASE=https://your-api-domain.com
NEXT_PUBLIC_API_KEY=production_api_key
```

### Static Hosting

The app supports static hosting on:
- Vercel (recommended)
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## Customization

### Themes

Modify `tailwind.config.js` for:
- Custom color schemes
- Typography scales
- Component variants
- Animation timing

### Components

Extend functionality by:
- Adding new page layouts
- Creating custom components
- Implementing additional features
- Integrating new API endpoints

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check `NEXT_PUBLIC_API_BASE` URL
   - Verify backend is running
   - Review CORS configuration

2. **Data Not Loading**
   - Check browser console
   - Verify API endpoints
   - Test API directly

3. **Build Errors**
   - Check TypeScript types
   - Verify imports
   - Review dependencies

4. **Styling Issues**
   - Clear Tailwind cache
   - Check class names
   - Verify responsive breakpoints

### Debug Commands

```bash
# Check environment
npm run build && npm run start

# Type checking
npx tsc --noEmit

# Linting
npm run lint

# Bundle analysis
npm run build
npx @next/bundle-analyzer
```