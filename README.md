# Trigger Monitor UI (Website Demo)

A polished website-style UI for the **Deal Trigger Deterioration Monitor** concept.

- Includes: landing page, live demo dashboard, pricing, security section
- Uses **mock data** (swap in real API later)
- Built with **Vite + React + Tailwind**, plus Recharts + Framer Motion

## Quickstart

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Where the UI lives

- Main component: `src/components/TriggerMonitorWebsiteDemo.jsx`
- Entry: `src/App.jsx`
- Basic UI primitives (button, card, etc.): `src/components/ui/...`

## Customizing

### Update pricing
Edit the `tiers` array in `src/components/TriggerMonitorWebsiteDemo.jsx`.

### Replace mock data
Search for `const MOCK` in `src/components/TriggerMonitorWebsiteDemo.jsx`.

### Hook up buttons
Buttons like **Request access** / **Request a pilot** are placeholders.
Replace with:
- a link to your calendar tool
- a contact form
- or a simple `mailto:`

## Deploy

You can deploy this anywhere that hosts static sites:

```bash
npm run build
```

Output is in `dist/`.
