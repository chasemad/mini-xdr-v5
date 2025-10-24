# Documents 08-10: Support, Licensing & UX

---

## 08: Support & Operations

**Solo Reality:** You ARE the support team  
**Priority:** P0 (Customers need help)  
**Time:** Ongoing + 2 weeks setup

### Critical Components

**Support Ticketing:**
- Use: Intercom ($74/month) or Plain (open source)
- Set up: Email → ticket conversion
- Response SLA: 24 hours (starter), 4 hours (enterprise)

**Knowledge Base:**
- Tool: GitBook or Notion (free tier)
- Must-have docs:
  - Getting started guide
  - Integration setup guides
  - Troubleshooting common issues
  - API documentation
  - Video tutorials (Loom)

**On-Call as Solo Dev:**
- Tools: PagerDuty free tier or Opsgenie
- Set up: Critical alerts only (P1 incidents)
- Reasonable hours: 9 AM - 9 PM, emergency only overnight
- Vacation: Set expectations with customers

**Customer Success:**
- Onboarding checklist (email automation)
- Weekly check-in for first month
- Monthly usage reports
- Quarterly business reviews (enterprise only)

### Quick Setup (Week 1)
- [ ] Set up Intercom or support email
- [ ] Create 10 essential help docs
- [ ] Set up canned responses
- [ ] Create status page (statuspage.io free tier)
- [ ] Write SLA policy document

### Quick Setup (Week 2)
- [ ] Create onboarding email sequence
- [ ] Build customer health dashboard
- [ ] Set up Net Promoter Score (NPS) survey
- [ ] Document escalation procedures
- [ ] Create emergency runbook

**Cost:** $100-200/month for tools

---

## 09: Licensing & Commercialization

**Current:** MIT (fully open)  
**Target:** Dual license (open core + commercial)  
**Priority:** P1 (Before first paying customer)  
**Time:** 1-2 weeks + lawyer ($3K-5K)

### Pricing Strategy (Recommended)

**Free Tier (Community Edition)**
- Single organization
- Up to 5 users
- 1,000 events/day
- Community support only
- Open source version

**Starter: $99/month**
- Up to 20 users
- 10,000 events/day
- Email support
- Basic integrations (3)
- 99% uptime SLA

**Professional: $499/month**
- Up to 100 users
- 100,000 events/day
- Priority support
- All integrations
- SSO (1 provider)
- 99.5% uptime SLA

**Enterprise: Custom ($2K-10K/month)**
- Unlimited users
- Unlimited events
- 24/7 phone support
- Custom integrations
- Multi-SSO
- On-premise option
- 99.9% uptime SLA
- Dedicated success manager

### Legal Documents Needed

**Week 1: Core Documents**
- [ ] Terms of Service (use Termly template, lawyer review: $1K)
- [ ] Privacy Policy (GDPR compliant, lawyer review: $1K)
- [ ] Acceptable Use Policy (standard template, $500)
- [ ] Data Processing Agreement (for GDPR, $1K)

**Week 2: Commercial**
- [ ] Master Services Agreement (MSA) - $2K
- [ ] Service Level Agreement (SLA) - included in MSA
- [ ] Order Form template - $500
- [ ] BAA for HIPAA (if applicable) - $1K

**Total Legal:** $3K-7K depending on complexity

### License Model

**Open Core:**
```
Mini-XDR Community Edition (MIT License)
- Core detection engine
- Basic ML models
- Single-tenant only
- Community support

Mini-XDR Enterprise (Commercial License)
- Everything in Community
- + Multi-tenancy
- + SSO/SAML
- + Advanced integrations
- + Enterprise support
- + SLA guarantees
```

**File:** Add `LICENSE-COMMERCIAL.md` to repo

### Payment Processing
- Stripe (2.9% + $0.30 per transaction)
- Monthly subscriptions with annual discount (20% off)
- Usage-based billing for enterprise
- Net 30 payment terms for enterprise contracts

### Quick Setup
- [ ] Choose pricing model
- [ ] Set up Stripe account
- [ ] Create pricing page on website
- [ ] Draft commercial license
- [ ] Get lawyer review
- [ ] Set up invoicing
- [ ] Create self-serve signup flow

**Cost:** Legal $3K-7K, Stripe fees 3%, accounting ~$500/month

---

## 10: User Experience & Accessibility

**Current:** Desktop web only, English only  
**Target:** Mobile-responsive, accessible, i18n-ready  
**Priority:** P2 (Nice to have, not blocker)  
**Time:** 3-4 weeks

### Mobile Responsiveness (Already 80% There)

**Check:** Open site on phone - frontend/Next.js is likely responsive
**Fix:** Any broken layouts in:
- Incident table (make scrollable)
- 3D globe (show 2D map on mobile)
- Agent chat (full screen on mobile)

```typescript
// Use Tailwind responsive classes
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
  {/* Responsive grid */}
</div>

// Mobile navigation
const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
```

**Week 1:**
- [ ] Test on iPhone, Android, tablet
- [ ] Fix broken layouts
- [ ] Make tables horizontally scrollable
- [ ] Optimize images for mobile
- [ ] Test touch interactions

### WCAG 2.1 AA Compliance (Accessibility)

**Critical (Week 2):**
- [ ] All images have alt text
- [ ] Proper heading hierarchy (h1, h2, h3)
- [ ] Keyboard navigation works
- [ ] Color contrast ratio ≥ 4.5:1
- [ ] Form labels present
- [ ] Focus indicators visible

**Tools:**
- axe DevTools (browser extension)
- WAVE (web accessibility evaluator)
- Lighthouse (Chrome DevTools)

**Quick fixes:**
```tsx
// Good: Semantic HTML
<button>Close</button>

// Bad: Non-semantic
<div onClick={handleClose}>Close</div>

// Good: ARIA labels
<button aria-label="Close incident">×</button>

// Good: Skip navigation
<a href="#main-content" className="skip-link">
  Skip to main content
</a>
```

### Internationalization (i18n) - FUTURE

**NOT needed for MVP, but prepare:**
```bash
npm install next-intl
```

**When you expand to EU/Asia:**
- Extract all strings to translation files
- Support at minimum: English, French, German, Spanish
- Date/time formatting per locale
- Currency formatting

**Cost to add later:** 2-3 weeks + $2K for professional translation

### Dark Mode (High Value, Low Effort)

**Week 3:**
```typescript
// Use next-themes
npm install next-themes

// Add to app/layout.tsx
import { ThemeProvider } from 'next-themes'

export default function RootLayout({ children }) {
  return (
    <html suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class">
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}

// Use Tailwind dark mode
<div className="bg-white dark:bg-gray-900 text-black dark:text-white">
```

### Native Mobile Apps - DON'T BUILD YET

**When to build:** After 100+ paying customers requesting it  
**Alternative:** Progressive Web App (PWA) is good enough

```typescript
// Convert to PWA (Week 4)
// Add to app/manifest.json
{
  "name": "Mini-XDR",
  "short_name": "XDR",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

### Solo Priority Order
1. **Week 1:** Mobile responsive (must-have)
2. **Week 2:** Accessibility basics (compliance)
3. **Week 3:** Dark mode (user request)
4. **Week 4:** PWA (mobile app alternative)
5. **Later:** i18n when expanding internationally
6. **Much Later:** Native apps when PMF proven

**Cost:** $0 (all free tools and libraries)

---

## What NOT to Do (Solo Developer)

### Don't Build (Yet):
- ❌ Native iOS/Android apps (use PWA)
- ❌ Desktop apps (Electron) - web is fine
- ❌ Custom mobile UI - responsive web works
- ❌ Full i18n before international customers
- ❌ Custom support portal - use Intercom
- ❌ Custom billing - use Stripe
- ❌ Video chat support - email/chat is enough

### DO Build:
- ✅ Excellent documentation
- ✅ Fast email/chat support
- ✅ Self-serve onboarding
- ✅ Clear pricing
- ✅ Status page
- ✅ Mobile-responsive web UI
- ✅ Basic accessibility

### Automate Everything:
- Onboarding emails (Intercom sequences)
- Invoice generation (Stripe)
- Usage reports (scheduled Python script)
- Status updates (statuspage.io)
- Common support questions (chatbot/canned responses)

---

## Summary: Weeks 1-4

**Week 1: Support Setup**
- Intercom or support email
- 10 help articles
- Status page
- On-call alerting

**Week 2: Licensing**
- Pricing finalized
- Legal docs drafted
- Stripe configured
- Self-serve signup

**Week 3: UX Polish**
- Mobile responsiveness fixes
- Accessibility audit
- Dark mode implementation
- PWA setup

**Week 4: Operations**
- Customer health monitoring
- Automated reporting
- Onboarding automation
- Documentation updates

**Total Investment:** 
- Time: 4 weeks
- Money: $3K-7K legal + $100-200/month tools
- Outcome: Ready to onboard paying customers

---

**Next:** `11_DEVOPS_CICD.md` and `12_REGULATORY_INDUSTRY_STANDARDS.md`


