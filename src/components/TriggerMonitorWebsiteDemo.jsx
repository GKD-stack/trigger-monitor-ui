import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import {
  Search,
  Bell,
  ShieldAlert,
  TrendingDown,
  Gauge,
  ChevronRight,
  Calendar,
  Filter,
  Download,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sparkles,
  Layers,
  LineChart as LineChartIcon,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card.jsx";
import { Button } from "./ui/button.jsx";
import { Badge } from "./ui/badge.jsx";
import { Input } from "./ui/input.jsx";
import { Separator } from "./ui/separator.jsx";
import { Progress } from "./ui/progress.jsx";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table.jsx";
import demoData from "../../out/trigger_monitor_demo.json";

/**
 * Trigger Monitor UI (Website Demo)
 * - Vite + React + Tailwind
 * - No backend required (mock data)
 * - Swap mock with API responses later
 */

const fmtPct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : "—");
const fmtNum = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const todayLabel = () =>
  new Date().toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });

function riskLabel(score) {
  if (score >= 0.75) return { label: "Red", icon: XCircle, tone: "destructive" };
  if (score >= 0.45) return { label: "Yellow", icon: AlertTriangle, tone: "secondary" };
  return { label: "Green", icon: CheckCircle2, tone: "outline" };
}

function percentRankLabel(p) {
  if (p >= 0.85) return "Severe";
  if (p >= 0.70) return "Moderate";
  return "Normal";
}

function formatMetricValue(metric, value, threshold) {
  const label = String(metric || "").toLowerCase();
  const hasPct = label.includes("%") || label.includes("percent");
  const nums = [value, threshold].filter((v) => Number.isFinite(v));
  const looksPct = nums.length ? nums.every((v) => Math.abs(v) <= 1) : false;
  return hasPct || looksPct ? fmtPct(value) : fmtNum(value);
}

// Legacy mock data removed in favor of SEC-fed demo data.
const DEMO = demoData;

function Pill({ tone = "outline", children, className = "" }) {
  return (
    <Badge variant={tone} className={`rounded-full px-2.5 py-1 text-xs ${className}`}>
      {children}
    </Badge>
  );
}

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <Card className="rounded-2xl">
      <CardContent className="p-5 min-h-[120px] flex items-center">
        <div className="flex w-full items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">{label}</div>
            <div className="text-2xl font-semibold tracking-tight">{value}</div>
            {sub ? <div className="text-xs text-muted-foreground">{sub}</div> : null}
          </div>
          <div className="h-10 w-10 rounded-2xl bg-muted flex items-center justify-center">
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SectionTitle({ kicker, title, desc, icon: Icon }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {Icon ? <Icon className="h-4 w-4" /> : null}
        <span className="uppercase tracking-wider">{kicker}</span>
      </div>
      <div className="text-2xl md:text-3xl font-semibold tracking-tight">{title}</div>
      {desc ? <div className="text-sm md:text-base text-muted-foreground max-w-2xl">{desc}</div> : null}
    </div>
  );
}

function NativeSelect({ value, onChange, options, className = "" }) {
  return (
    <div className={`relative ${className}`}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-10 w-full rounded-2xl border border-border bg-background px-3 pr-10 text-sm outline-none focus:ring-2 focus:ring-offset-2"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <Filter className="h-4 w-4 text-muted-foreground absolute right-3 top-3 pointer-events-none" />
    </div>
  );
}

function Nav({ active, onNavigate }) {
  const items = [
    { key: "product", label: "Product" },
    { key: "demo", label: "Live Demo" },
    { key: "pricing", label: "Pricing" },
    { key: "security", label: "Security" },
  ];

  return (
    <div className="sticky top-0 z-40 backdrop-blur bg-white/70 border-b border-border">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-9 w-9 rounded-2xl bg-muted flex items-center justify-center">
            <Gauge className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <div className="font-semibold">Trigger Monitor</div>
            <div className="text-xs text-muted-foreground">Deal deterioration radar</div>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2">
          {items.map((it) => (
            <Button
              key={it.key}
              variant={active === it.key ? "secondary" : "ghost"}
              className="rounded-2xl"
              onClick={() => onNavigate(it.key)}
            >
              {it.label}
            </Button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            className="rounded-2xl hidden sm:inline-flex"
            href="/reports/trigger-monitor-sample-report.pdf"
            download
          >
            <Download className="h-4 w-4 mr-2" /> Sample report
          </Button>
          <Button className="rounded-2xl" onClick={() => onNavigate("request")}>
            Request access <ChevronRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function Hero({ onNavigate }) {
  return (
    <div className="mx-auto max-w-6xl px-4 py-12 md:py-16">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center"
      >
        <div className="space-y-5">
          <Pill className="w-fit">Explainable surveillance • Email + dashboard • No model black boxes</Pill>
          <div className="text-4xl md:text-5xl font-semibold tracking-tight leading-tight">
            Know which deals are cracking <span className="text-muted-foreground">before</span> triggers trip.
          </div>
          <div className="text-base md:text-lg text-muted-foreground max-w-xl">
            Trigger Monitor watches OC/IC cushions, collateral drift, and macro regimes across your book. It ranks what’s worsening,
            explains why, and keeps your team focused.
          </div>
          <div className="flex flex-col sm:flex-row gap-3">
            <Button className="rounded-2xl" onClick={() => onNavigate("demo")}>
              View live demo <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
            <Button variant="secondary" className="rounded-2xl" onClick={() => onNavigate("pricing")}>
              See pricing
            </Button>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-2"><ShieldAlert className="h-4 w-4" /> Private by default</div>
            <div className="flex items-center gap-2"><Calendar className="h-4 w-4" /> Weekly or monthly</div>
            <div className="flex items-center gap-2"><Sparkles className="h-4 w-4" /> Explainable scoring</div>
          </div>
        </div>

        <Card className="rounded-3xl overflow-hidden">
          <CardHeader className="pb-0 pt-6">
            <CardTitle className="text-base">This week’s deterioration snapshot</CardTitle>
            <CardDescription>As of {todayLabel()}</CardDescription>
          </CardHeader>
          <CardContent className="p-5 pt-6">
            <div className="grid grid-cols-2 gap-3">
              <Stat icon={TrendingDown} label="Flagged deals" value={`${DEMO.portfolio.flagged}/${DEMO.portfolio.deals}`} sub={`${DEMO.portfolio.red} red • ${DEMO.portfolio.yellow} yellow`} />
              <Stat icon={Bell} label="New alerts" value={`${DEMO.alerts.length}`} sub="Since last run" />
            </div>
            <Separator className="my-4" />
            <div className="space-y-3">
              {DEMO.alerts.slice(0, 3).map((a) => (
                <div key={a.ts} className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Pill tone={a.severity === "red" ? "destructive" : "secondary"}>
                        {a.severity.toUpperCase()}
                      </Pill>
                      <div className="text-sm font-medium">{a.title}</div>
                    </div>
                    <div className="text-xs text-muted-foreground">{a.detail}</div>
                  </div>
                  <div className="text-[11px] text-muted-foreground whitespace-nowrap">{a.ts}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function FeatureGrid() {
  const features = [
    { icon: Layers, title: "Positions-first", desc: "Upload a simple positions list. We map to deal IDs/CUSIPs and monitor only what you own." },
    { icon: LineChartIcon, title: "Cushion time series", desc: "Track OC/IC and other trigger distances over time. Level + trend + volatility in one view." },
    { icon: ShieldAlert, title: "Explainable alerts", desc: "No black boxes—every score comes with a plain-English rationale and the underlying drivers." },
    { icon: Bell, title: "Email + dashboard", desc: "Weekly or monthly ranked updates. Optional portal for drill-down and audit trail." },
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <SectionTitle
        kicker="How it works"
        title="A focused deterioration radar"
        desc="We’re not trying to price tranches or run full cashflow models. We highlight where structural protection is disappearing faster than it should—so you can re-underwrite early."
        icon={Gauge}
      />
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {features.map((f) => (
          <Card key={f.title} className="rounded-2xl">
            <CardContent className="p-6 pt-7 space-y-4">
              <div className="h-10 w-10 rounded-2xl bg-muted flex items-center justify-center mt-1">
                <f.icon className="h-5 w-5" />
              </div>
              <div className="font-semibold">{f.title}</div>
              <div className="text-sm text-muted-foreground">{f.desc}</div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function DemoDashboard({ onRequest }) {
  const [query, setQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");
  const deals = useMemo(() => DEMO?.deals ?? [], []);
  if (!deals.length) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-12">
        <Card className="rounded-2xl">
          <CardContent className="p-6">
            <div className="text-sm text-muted-foreground">No demo data loaded yet.</div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const rows = useMemo(() => {
    const flat = deals.flatMap((d) =>
      d.triggers.map((t) => ({
        dealId: d.dealId,
        cusip: d.cusip,
        collateral: d.collateral,
        tranche: d.tranche,
        geo: d.geo,
        macroTheme: d.macro.theme,
        macroPct: d.macro.percentile,
        macroRegime: d.macro.source ? percentRankLabel(d.macro.percentile) : "Normal",
        ...t,
      }))
    );

    const filtered = flat.filter((r) => {
      const q = query.trim().toLowerCase();
      const hit = !q
        ? true
        : [r.dealId, r.cusip, r.collateral, r.tranche, r.metric, r.triggerId]
            .join(" ")
            .toLowerCase()
            .includes(q);
      const regime = r.macroRegime.toLowerCase();
      const passRisk = riskFilter === "all" ? true : regime === riskFilter;
      return hit && passRisk;
    });

    return filtered.sort((a, b) => b.score - a.score);
  }, [deals, query, riskFilter]);

  const [selectedDealId, setSelectedDealId] = useState(deals[0].dealId);
  const selected = useMemo(() => deals.find((d) => d.dealId === selectedDealId) ?? deals[0], [selectedDealId, deals]);
  const macroRegime = selected.macro?.source ? percentRankLabel(selected.macro.percentile) : "Normal";
  const collateralMetrics = selected.collateralMetrics ?? [];
  const cushionSeries = selected.cushionSeries ?? [];
  const dqSeries = selected.dqSeries ?? [];
  const cushionSample = cushionSeries[0] ?? {};

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <div className="flex items-start justify-between gap-4 flex-col md:flex-row">
        <SectionTitle
          kicker="Live demo"
          title="Ranked trigger deterioration"
          desc="This is what a PM sees: the deals where cushion is shrinking fastest, with an explainable rationale and quick drill-down."
          icon={Bell}
        />
        <div className="flex items-center gap-2">
          <Button variant="secondary" className="rounded-2xl">
            <Download className="h-4 w-4 mr-2" /> Export
          </Button>
          <Button className="rounded-2xl" onClick={onRequest}>
            Request a pilot <ChevronRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="rounded-2xl lg:col-span-2">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between gap-3 flex-col md:flex-row">
              <div>
                <CardTitle className="text-base">Most at-risk triggers</CardTitle>
                <CardDescription>Sorted by risk score (explainable rule-based scoring)</CardDescription>
              </div>
              <div className="flex gap-2 w-full md:w-auto">
                <div className="relative w-full md:w-72">
                  <Search className="h-4 w-4 text-muted-foreground absolute left-3 top-3" />
                  <Input
                    className="pl-9 rounded-2xl"
                    placeholder="Search deal, CUSIP, metric…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
                <NativeSelect
                  value={riskFilter}
                  onChange={setRiskFilter}
                  options={[
                    { value: "all", label: "All" },
                    { value: "normal", label: "Normal" },
                    { value: "moderate", label: "Moderate" },
                    { value: "severe", label: "Severe" },
                  ]}
                  className="w-[150px]"
                />
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Deal</TableHead>
                  <TableHead>Tranche</TableHead>
                  <TableHead>Trigger</TableHead>
                  <TableHead className="text-right">Cushion</TableHead>
                  <TableHead className="text-right">Δ 3m</TableHead>
                  <TableHead>Macro</TableHead>
                  <TableHead className="text-right">Risk</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => {
                  const rl = riskLabel(r.score);
                  return (
                    <TableRow key={`${r.dealId}-${r.triggerId}`} className="cursor-pointer" onClick={() => setSelectedDealId(r.dealId)}>
                      <TableCell>
                        <div className="font-medium">{r.dealId}</div>
                        <div className="text-xs text-muted-foreground">{r.collateral} • {r.cusip}</div>
                      </TableCell>
                      <TableCell>
                        <div className="font-medium">{r.tranche}</div>
                        <div className="text-xs text-muted-foreground">{r.geo}</div>
                      </TableCell>
                      <TableCell>
                        <div className="font-medium">{r.metric}</div>
                        <div className="text-xs text-muted-foreground">
                          {r.direction} {formatMetricValue(r.metric, r.threshold, r.threshold)} • current{" "}
                          {formatMetricValue(r.metric, r.current, r.threshold)}
                        </div>
                      </TableCell>
                      <TableCell className="text-right"><div className="font-medium">{fmtPct(r.cushion)}</div></TableCell>
                      <TableCell className="text-right"><div className="font-medium">{fmtPct(r.change3m)}</div></TableCell>
                      <TableCell>
                        <Pill tone={r.macroRegime === "Severe" ? "destructive" : r.macroRegime === "Moderate" ? "secondary" : "outline"}>{r.macroRegime}</Pill>
                        <div className="text-xs text-muted-foreground mt-1">{r.macroTheme}</div>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <rl.icon className="h-4 w-4" />
                          <span className="font-medium">{Math.round(r.score * 100)}</span>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Selected deal</CardTitle>
            <CardDescription>{selected.dealId} • {selected.tranche}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">Macro regime</div>
                <div className="text-xs text-muted-foreground">{selected.macro.series}</div>
              </div>
              <Pill tone={macroRegime === "Severe" ? "destructive" : macroRegime === "Moderate" ? "secondary" : "outline"}>
                {macroRegime} ({Math.round(selected.macro.percentile * 100)}th)
              </Pill>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium">Deal risk</div>
                <div className="text-xs text-muted-foreground">Max trigger score</div>
              </div>
              <Progress value={Math.round(Math.max(...selected.triggers.map((t) => t.score)) * 100)} />
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">Why it’s flagged</div>
              <div className="text-sm text-muted-foreground leading-relaxed">{selected.explanation}</div>
            </div>
            <Separator />
            <div className="grid grid-cols-2 gap-3">
              {collateralMetrics.map((m) => (
                <div key={m.name} className="rounded-2xl border border-border p-3">
                  <div className="text-xs text-muted-foreground">{m.name}</div>
                  <div className="mt-1 flex items-baseline justify-between">
                    <div className="text-lg font-semibold">{fmtPct(m.cur)}</div>
                    <div className="text-xs text-muted-foreground">{m.chg >= 0 ? "+" : ""}{fmtPct(m.chg)}</div>
                  </div>
                </div>
              ))}
            </div>
            <Button
              variant="secondary"
              className="rounded-2xl w-full"
              onClick={() => window.open(`/deal-report.html?deal=${encodeURIComponent(selected.dealId)}`, "_blank")}
            >
              View deal report <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Trigger cushion over time</CardTitle>
            <CardDescription>OC/IC cushion (distance to threshold)</CardDescription>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={cushionSeries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="m" tickMargin={8} />
                <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={(v) => fmtPct(v)} />
                {cushionSample.oc !== undefined ? <Line type="monotone" dataKey="oc" strokeWidth={2} dot={false} /> : null}
                {cushionSample.ic !== undefined ? <Line type="monotone" dataKey="ic" strokeWidth={2} dot={false} /> : null}
                {cushionSample.dq !== undefined ? <Line type="monotone" dataKey="dq" strokeWidth={2} dot={false} /> : null}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Collateral deterioration</CardTitle>
            <CardDescription>60+ delinquency rate (trend)</CardDescription>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dqSeries} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="m" tickMargin={8} />
                <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={(v) => fmtPct(v)} />
                <Bar dataKey="dq60" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8">
        <Card className="rounded-2xl">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">What you get in production</CardTitle>
            <CardDescription>Same UI, but backed by your data pipeline and scheduled monitoring</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-border p-4">
              <div className="font-semibold">Weekly ranking</div>
              <div className="text-sm text-muted-foreground mt-1">Top deals where cushions are shrinking fastest, across your positions.</div>
            </div>
            <div className="rounded-2xl border border-border p-4">
              <div className="font-semibold">Explainable drivers</div>
              <div className="text-sm text-muted-foreground mt-1">Each flag includes a rationale: cushion, trend, volatility, and macro regime.</div>
            </div>
            <div className="rounded-2xl border border-border p-4">
              <div className="font-semibold">Audit trail</div>
              <div className="text-sm text-muted-foreground mt-1">Snapshots by period for compliance and post-mortems.</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Pricing({ onRequest }) {
  const tiers = [
    { name: "Starter", price: "$350/mo", tag: "Solo / emerging", items: ["Up to 15 deals", "Monthly email summary", "Web dashboard (read-only)", "Explainable scoring", "Standard trigger set"] },
    { name: "Pilot", price: "$750/mo", tag: "Most common", items: ["Up to 40 deals", "Weekly email summary", "Web dashboard (read-only)", "Explainable scoring", "1 custom metric mapping"] },
    { name: "Core", price: "$1,200/mo", tag: "Teams", items: ["Up to 80 deals", "Weekly + monthly reports", "Custom triggers/metrics", "SLA + monitoring", "Dedicated onboarding"] },
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <SectionTitle kicker="Pricing" title="Simple subscription, priced by deal count" desc="Start small. Expand coverage as your book grows. Cancel anytime." icon={Sparkles} />
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {tiers.map((t) => (
          <Card key={t.name} className="rounded-2xl">
            <CardContent className="p-7 pt-8 space-y-4 min-h-[320px]">
              <div className="flex items-center justify-between">
                <div className="text-lg font-semibold">{t.name}</div>
                <Pill tone={t.name === "Pilot" ? "secondary" : "outline"}>{t.tag}</Pill>
              </div>
              <div className="text-3xl font-semibold tracking-tight">{t.price}</div>
              <div className="space-y-2">
                {t.items.map((i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 mt-0.5" />
                    <div className="text-muted-foreground">{i}</div>
                  </div>
                ))}
              </div>
              <Button className="rounded-2xl w-full" onClick={onRequest}>Request access</Button>
              <div className="text-xs text-muted-foreground">No trading. No execution. Monitoring only.</div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function Security() {
  const bullets = [
    { title: "Private-by-default", desc: "Run in your VPC or ours. Data is scoped per client; no cross-client learning required." },
    { title: "Explainable outputs", desc: "Every flag is traceable to a metric and threshold—designed for IC, risk, and compliance review." },
    { title: "Minimal permissions", desc: "Ingest your parsed trustee/SEC outputs (or a secure bucket). No need for trading system access." },
  ];
  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <SectionTitle kicker="Security" title="Built to be forwarded to risk and compliance" desc="We keep the surface area small: read-only monitoring, clear audit trails, and deploy options that fit your policies." icon={ShieldAlert} />
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {bullets.map((b) => (
          <Card key={b.title} className="rounded-2xl">
            <CardContent className="p-7 pt-8 space-y-3 min-h-[220px] text-center">
              <div className="font-semibold">{b.title}</div>
              <div className="text-sm text-muted-foreground">{b.desc}</div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function RequestAccess() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <SectionTitle
        kicker="Request access"
        title="Get a pilot set up"
        desc="Submit your details and we’ll follow up with a tailored demo and data intake checklist."
        icon={Sparkles}
      />
      <div className="mt-6">
        <Card className="rounded-2xl overflow-hidden">
          <CardContent className="p-0">
            <iframe
              className="w-full min-h-[560px] border-0"
              src="https://airtable.com/embed/appkMThBbT1iCic0M/pagsoMnqr2lnsUgXJ/form"
              title="Request access form"
              frameBorder="0"
              width="100%"
              height="533"
              loading="lazy"
              style={{ background: "transparent" }}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function TriggerMonitorWebsiteDemo() {
  const [active, setActive] = useState("product");
  const handleNavigate = (key) => {
    setActive(key);
    const target = document.getElementById(`section-${key}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Nav active={active} onNavigate={handleNavigate} />
      <section id="section-product" className="bg-gradient-to-b from-[#f3efe9] via-[#f7f4ef] to-white">
        <Hero onNavigate={handleNavigate} />
        <FeatureGrid />
        <div className="mx-auto max-w-6xl px-4 pb-12">
          <Card className="rounded-3xl">
            <CardContent className="p-6 md:p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <div className="space-y-2">
                  <div className="text-lg font-semibold">Designed for small funds</div>
                  <div className="text-sm text-muted-foreground">A clean UI to show in meetings, plus automated monitoring in production.</div>
                </div>
                <div className="flex flex-col sm:flex-row gap-3 justify-end">
                  <Button variant="secondary" className="rounded-2xl" onClick={() => handleNavigate("demo")}>Open demo dashboard</Button>
                  <Button className="rounded-2xl" onClick={() => handleNavigate("pricing")}>Pricing</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      <section id="section-demo" className="bg-white">
        <DemoDashboard onRequest={() => handleNavigate("request")} />
      </section>
      <section id="section-pricing" className="bg-[#f7f4ef]">
        <Pricing onRequest={() => handleNavigate("request")} />
      </section>
      <section id="section-security" className="bg-[#f2f5f4]">
        <Security />
      </section>
      <section id="section-request" className="bg-white">
        <RequestAccess />
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-6xl px-4 py-8 flex flex-col md:flex-row gap-3 items-start md:items-center justify-between">
          <div className="text-sm text-muted-foreground">© {new Date().getFullYear()} Trigger Monitor • Deal deterioration radar</div>
          <div className="text-xs text-muted-foreground">Demo uses mock data. Production connects to your trustee/SEC parsing outputs.</div>
        </div>
      </footer>
    </div>
  );
}
