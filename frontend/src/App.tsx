import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  fetchModelPrediction,
  SUGGESTED_QUERIES,
  type LegalResult,
  type ModelResponse,
} from "@/lib/judicia-data";

type Phase = "idle" | "loading" | "results" | "empty" | "error";

function App() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [modelResponse, setModelResponse] = useState<ModelResponse | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        setIsBackendHealthy(true);
      } else {
        setIsBackendHealthy(false);
      }
    } catch {
      setIsBackendHealthy(false);
    }
  }

  async function run(raw: string) {
    const q = raw.trim();
    if (!q) return;
    setQuery(q);
    setSubmitted(q);
    setPhase("loading");
    setModelResponse(null);

    if (timer.current) clearTimeout(timer.current);

    try {
      const data = await fetchModelPrediction(q);
      setModelResponse(data);
      if (data.results && data.results.length > 0) {
        setPhase("results");
      } else {
        setPhase("empty");
      }
    } catch (err) {
      console.error(err);
      setPhase("error");
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    run(query);
  }

  const compact = phase !== "idle";

  return (
    <div className="min-h-screen bg-background">
      <Header isHealthy={isBackendHealthy} onRecheckHealth={checkHealth} />

      <main className="mx-auto w-full max-w-5xl px-5 pb-24 sm:px-8">
        <section
          className={`rule-grid -mx-5 px-5 sm:-mx-8 sm:px-8 transition-all duration-500 ${
            compact ? "pt-8 pb-6 sm:pt-10" : "pt-16 pb-14 sm:pt-24 sm:pb-20"
          }`}
        >
          <div className="mx-auto max-w-3xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground shadow-sm">
              <span
                className={`h-2 w-2 rounded-full ${
                  isBackendHealthy ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              {isBackendHealthy
                ? "BERT Domain Classification Model Connected"
                : "Judicia Legal Domain Model"}
            </span>

            <h1
              className={`text-balance-tight mt-6 font-display font-semibold tracking-tight text-ink transition-all duration-500 ${
                compact ? "text-4xl sm:text-5xl" : "text-6xl sm:text-7xl"
              }`}
            >
              Judicia AI
            </h1>

            {!compact && (
              <p className="mx-auto mt-5 max-w-xl text-balance-tight text-base leading-relaxed text-muted-foreground sm:text-lg">
                Enter your legal query below. Your query will be evaluated live by our custom fine-tuned BERT model (<code>judicia-domain-model</code>) to predict legal domain classification and retrieve relevant statutory provisions & precedents.
              </p>
            )}

            <form onSubmit={onSubmit} className="mt-8">
              <div className="surface-plate group flex flex-col gap-3 rounded-2xl p-3 focus-within:border-accent sm:flex-row sm:items-center sm:rounded-full sm:p-2 sm:pl-5 shadow-lg">
                <SearchGlyph />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. Someone hacked my account or My employer terminated me without salary"
                  aria-label="Search legal question"
                  className="min-w-0 flex-1 bg-transparent px-2 py-3 text-[15px] text-foreground outline-none placeholder:text-muted-foreground sm:px-0"
                />
                <button
                  type="submit"
                  disabled={!query.trim() || phase === "loading"}
                  className="ink-panel inline-flex h-12 items-center justify-center gap-2 rounded-xl px-7 text-sm font-semibold tracking-wide transition-all duration-200 hover:brightness-125 disabled:cursor-not-allowed disabled:opacity-45 sm:rounded-full cursor-pointer"
                >
                  {phase === "loading" ? "Analyzing with Model..." : "Ask Model"}
                </button>
              </div>
            </form>

            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {SUGGESTED_QUERIES.slice(0, compact ? 2 : 4).map((s) => (
                <button
                  key={s}
                  onClick={() => run(s)}
                  className="rounded-full border border-border bg-card/80 px-3.5 py-1.5 text-xs text-muted-foreground transition-all hover:border-accent hover:text-ink cursor-pointer"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-8">
          {phase === "idle" && <IdleState />}
          {phase === "loading" && <LoadingState />}
          {phase === "empty" && <NoResults query={submitted} onReset={() => setPhase("idle")} />}
          {phase === "results" && modelResponse && (
            <Results query={submitted} modelResponse={modelResponse} />
          )}
        </section>
      </main>

      <Footer />
    </div>
  );
}

function Header({
  isHealthy,
  onRecheckHealth,
}: {
  isHealthy: boolean | null;
  onRecheckHealth: () => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-border/70 bg-background/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-5 py-4 sm:px-8">
        <div className="flex items-center gap-2.5">
          <Mark />
          <span className="font-display text-lg font-semibold tracking-tight text-ink">
            Judicia
          </span>
          <span className="rounded bg-secondary/80 px-2 py-0.5 font-mono text-[10px] uppercase text-muted-foreground">
            v1.0 ML
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onRecheckHealth}
            title="Click to check backend status"
            className="flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-accent hover:text-ink cursor-pointer"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isHealthy === true
                  ? "bg-emerald-500"
                  : isHealthy === false
                  ? "bg-amber-500"
                  : "bg-gray-400 animate-ping"
              }`}
            />
            {isHealthy === true
              ? "Model API: Connected"
              : isHealthy === false
              ? "Model API: Disconnected"
              : "Checking API..."}
          </button>
        </div>
      </div>
    </header>
  );
}

function Mark() {
  return (
    <span className="ink-panel flex h-8 w-8 items-center justify-center rounded-lg">
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-accent" aria-hidden="true">
        <path
          d="M12 3v18M5 8h14M7 8l-3 6a3 3 0 0 0 6 0L7 8Zm10 0-3 6a3 3 0 0 0 6 0l-3-6Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}

function SearchGlyph() {
  return (
    <svg viewBox="0 0 24 24" className="ml-2 h-5 w-5 shrink-0 text-muted-foreground sm:ml-0" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="m16 16 4.5 4.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function IdleState() {
  const pillars = [
    { k: "01", t: "BERT Domain Classifier", d: "Processes legal queries and predicts target legal domain (Cyber, Criminal, Property, Family, Labour, Civil, Consumer, Constitutional)." },
    { k: "02", t: "Confidence & Multi-Class Scoring", d: "Calculates softmax probability distributions across all 8 legal domain categories to quantify model certainty." },
    { k: "03", t: "Retrieved Legal Provisions", d: "Surfaces statutory provisions and precedents linked directly to the model's domain prediction." },
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {pillars.map((p) => (
        <article key={p.k} className="surface-plate rounded-2xl p-6">
          <span className="font-mono text-xs tracking-[0.2em] text-accent">{p.k}</span>
          <h2 className="mt-3 font-display text-lg font-semibold text-ink">{p.t}</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{p.d}</p>
        </article>
      ))}
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <div className="flex items-center justify-between">
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-accent animate-pulse">
          Evaluating Query with judicia-domain-model…
        </p>
      </div>
      <div className="surface-plate rounded-2xl p-6 border border-accent/30">
        <div className="skeleton-sheen h-4 w-48 rounded" />
        <div className="skeleton-sheen mt-4 h-6 w-3/4 rounded" />
        <div className="skeleton-sheen mt-3 h-3 w-full rounded" />
      </div>
      {[0, 1].map((i) => (
        <div key={i} className="surface-plate rounded-2xl p-6">
          <div className="skeleton-sheen h-3 w-24 rounded" />
          <div className="skeleton-sheen mt-4 h-5 w-2/3 rounded" />
          <div className="skeleton-sheen mt-3 h-3 w-full rounded" />
        </div>
      ))}
    </div>
  );
}

function NoResults({ query, onReset }: { query: string; onReset: () => void }) {
  return (
    <div className="surface-plate animate-rise rounded-2xl px-6 py-14 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-secondary">
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-muted-foreground" aria-hidden="true">
          <circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <path d="m16 16 4.5 4.5M8.5 11h5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      </div>
      <h2 className="mt-5 font-display text-xl font-semibold text-ink">
        No matches for "{query}"
      </h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-muted-foreground">
        Please try asking a complete legal question or describing your situation.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        <button
          onClick={onReset}
          className="rounded-full border border-border px-4 py-2 text-sm text-ink transition-colors hover:border-accent cursor-pointer"
        >
          Start over
        </button>
      </div>
    </div>
  );
}

function Results({
  query,
  modelResponse,
}: {
  query: string;
  modelResponse: ModelResponse;
}) {
  const { domain, confidence, probabilities, results, isModelLive } = modelResponse;
  const confPct = Math.round(confidence * 100);

  // Top probabilities array sorted descending
  const sortedProbs = Object.entries(probabilities)
    .map(([lbl, val]) => ({ label: lbl, score: val }))
    .sort((a, b) => b.score - a.score);

  return (
    <div className="space-y-6">
      {/* Model Inference Summary Banner */}
      <div className="rounded-2xl border border-accent/40 bg-card p-6 shadow-md">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="font-mono text-xs uppercase tracking-[0.16em] text-accent">
              ML Model Output Analysis
            </span>
            <h2 className="mt-1 font-display text-2xl font-bold text-ink">
              Predicted Domain: <span className="text-accent underline underline-offset-4">{domain}</span>
            </h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">
              Query: "{query}"
            </p>
          </div>
          <div className="flex flex-col items-end">
            <div className="flex items-center gap-2">
              <span className="font-mono text-3xl font-extrabold text-ink">{confPct}%</span>
              <span className="text-xs text-muted-foreground">Confidence</span>
            </div>
            <div className="mt-1 h-2 w-32 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${confPct}%`,
                  background: "linear-gradient(90deg, #d97706, #10b981)",
                }}
              />
            </div>
            <span className="mt-1 font-mono text-[10px] text-muted-foreground">
              {isModelLive ? "Live Inference: judicia-domain-model" : "Fallback Offline Mode"}
            </span>
          </div>
        </div>

        {/* Probability breakdown */}
        {sortedProbs.length > 1 && (
          <div className="mt-6 border-t border-border pt-4">
            <h4 className="font-mono text-xs uppercase tracking-[0.12em] text-muted-foreground mb-3">
              Model Class Distribution Breakdown:
            </h4>
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
              {sortedProbs.slice(0, 8).map((p) => {
                const percentage = Math.round(p.score * 100);
                const isTop = p.label === domain;
                return (
                  <div
                    key={p.label}
                    className={`rounded-xl border p-2.5 transition-all ${
                      isTop
                        ? "border-accent/80 bg-accent/10 shadow-sm"
                        : "border-border bg-background/50"
                    }`}
                  >
                    <div className="flex justify-between font-mono text-xs">
                      <span className={isTop ? "font-bold text-ink" : "text-muted-foreground"}>
                        {p.label}
                      </span>
                      <span className={isTop ? "font-bold text-accent" : "text-muted-foreground"}>
                        {percentage}%
                      </span>
                    </div>
                    <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${percentage}%`,
                          backgroundColor: isTop ? "var(--accent, #d97706)" : "#6b7280",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Retrieved Passages Header */}
      <div className="flex items-baseline justify-between border-b border-border pb-3">
        <h3 className="font-display text-lg font-semibold text-ink">
          {results.length} Relevant Statutory Provisions & Precedents
        </h3>
        <span className="font-mono text-xs text-muted-foreground">
          Matched under {domain}
        </span>
      </div>

      {/* Results Cards */}
      <div className="space-y-4">
        {results.map((r, i) => (
          <ResultCard key={r.id} result={r} index={i} />
        ))}
      </div>
    </div>
  );
}

function ResultCard({ result, index }: { result: LegalResult; index: number }) {
  const pct = Math.round(result.match * 100);
  return (
    <article
      className="surface-plate-interactive animate-rise rounded-2xl p-6"
      style={{ animationDelay: `${index * 70}ms` }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        <span className="rounded-md bg-secondary px-2.5 py-1 font-semibold text-ink-soft">{result.category}</span>
        <span>{result.jurisdiction}</span>
        <span aria-hidden="true">·</span>
        <span>{result.year}</span>
        <span className="ml-auto flex items-center gap-2 normal-case tracking-normal">
          <span className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
            <span
              className="block h-full rounded-full"
              style={{ width: `${pct}%`, background: "var(--gradient-brass, linear-gradient(90deg, #d97706, #10b981))" }}
            />
          </span>
          {pct}% relevance
        </span>
      </div>

      <h3 className="mt-4 font-display text-xl font-semibold leading-snug text-ink">
        {result.title}
      </h3>
      <p className="mt-1 text-sm text-muted-foreground font-mono">{result.citation}</p>

      <blockquote className="mt-4 border-l-2 border-accent pl-4 text-[15px] leading-relaxed text-ink-soft bg-card/30 py-1.5 rounded-r">
        {result.snippet}
      </blockquote>

      <div className="mt-4 flex flex-wrap gap-1.5">
        {result.passages.map((p) => (
          <span key={p} className="rounded-full border border-border bg-secondary/60 px-3 py-1 text-xs text-ink-soft font-mono">
            {p}
          </span>
        ))}
      </div>

      <div className="mt-5 flex flex-wrap items-end justify-between gap-4 border-t border-border pt-4">
        <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
          <span className="font-semibold text-ink">Model Attribution — </span>
          {result.why}
        </p>
      </div>
    </article>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border mt-16">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-2 px-5 py-8 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>Judicia AI · Powered by judicia-domain-model (BERT)</p>
        <p>Legal domain classification and statutory passage retrieval.</p>
      </div>
    </footer>
  );
}

export default App;