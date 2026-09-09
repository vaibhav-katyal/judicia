import { useRef, useState, type FormEvent } from "react";
import { searchCorpus, SUGGESTED_QUERIES, type LegalResult } from "@/lib/judicia-data";

type Phase = "idle" | "loading" | "results" | "empty";

function App() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [results, setResults] = useState<LegalResult[]>([]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function run(raw: string) {
    const q = raw.trim();
    if (!q) return;
    setQuery(q);
    setSubmitted(q);
    setPhase("loading");
    setResults([]);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const found = searchCorpus(q);
      setResults(found);
      setPhase(found.length ? "results" : "empty");
    }, 900);
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    run(query);
  }

  const compact = phase !== "idle";

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="mx-auto w-full max-w-5xl px-5 pb-24 sm:px-8">
        <section
          className={`rule-grid -mx-5 px-5 sm:-mx-8 sm:px-8 transition-all duration-500 ${
            compact ? "pt-10 pb-8 sm:pt-14" : "pt-16 pb-14 sm:pt-28 sm:pb-20"
          }`}
        >
          <div className="mx-auto max-w-3xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" />
              Retrieval-grounded legal research
            </span>

            <h1
              className={`text-balance-tight mt-6 font-display font-semibold tracking-tight text-ink transition-all duration-500 ${
                compact ? "text-4xl sm:text-5xl" : "text-6xl sm:text-7xl"
              }`}
            >
              Judicia
            </h1>

            {!compact && (
              <p className="mx-auto mt-5 max-w-xl text-balance-tight text-base leading-relaxed text-muted-foreground sm:text-lg">
                Ask in plain language. Judicia reads the statute book and reported
                judgments, then returns the passages that actually govern your
                question — each with its citation and why it matched.
              </p>
            )}

            <form onSubmit={onSubmit} className="mt-9">
              <div className="surface-plate group flex flex-col gap-3 rounded-2xl p-3 focus-within:border-accent sm:flex-row sm:items-center sm:rounded-full sm:p-2 sm:pl-5">
                <SearchGlyph />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. When is an electronic record admissible in evidence?"
                  aria-label="Search legal question"
                  className="min-w-0 flex-1 bg-transparent px-2 py-3 text-[15px] text-foreground outline-none placeholder:text-muted-foreground sm:px-0"
                />
                <button
                  type="submit"
                  disabled={!query.trim() || phase === "loading"}
                  className="ink-panel inline-flex h-12 items-center justify-center gap-2 rounded-xl px-6 text-sm font-semibold tracking-wide transition-all duration-200 hover:brightness-125 disabled:cursor-not-allowed disabled:opacity-45 sm:rounded-full"
                >
                  {phase === "loading" ? "Searching" : "Search"}
                </button>
              </div>
            </form>

            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {SUGGESTED_QUERIES.slice(0, compact ? 2 : 4).map((s) => (
                <button
                  key={s}
                  onClick={() => run(s)}
                  className="rounded-full border border-border bg-card/70 px-3.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-accent hover:text-ink"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-10">
          {phase === "idle" && <IdleState />}
          {phase === "loading" && <LoadingState />}
          {phase === "empty" && <NoResults query={submitted} onReset={() => setPhase("idle")} />}
          {phase === "results" && <Results query={submitted} results={results} />}
        </section>
      </main>

      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-border/70 bg-background/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-5 py-4 sm:px-8">
        <div className="flex items-center gap-2.5">
          <Mark />
          <span className="font-display text-lg font-semibold tracking-tight text-ink">
            Judicia
          </span>
        </div>
        <nav className="hidden items-center gap-7 text-sm text-muted-foreground sm:flex">
          <span className="transition-colors hover:text-ink">Corpus</span>
          <span className="transition-colors hover:text-ink">Citations</span>
          <span className="transition-colors hover:text-ink">Method</span>
        </nav>
      </div>
    </header>
  );
}

function Mark() {
  return (
    <span className="ink-panel flex h-8 w-8 items-center justify-center rounded-lg">
      <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
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
    { k: "01", t: "Grounded in the source", d: "Every answer points back to a provision or reported judgment — never an unsourced summary." },
    { k: "02", t: "Passage-level retrieval", d: "Judicia surfaces the operative paragraph, not the whole volume, so you read what matters." },
    { k: "03", t: "Explained relevance", d: "Each result states why it matched, so you can accept or discard it in seconds." },
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
      <p className="font-mono text-xs uppercase tracking-[0.18em] text-muted-foreground">
        Reading the corpus…
      </p>
      {[0, 1, 2].map((i) => (
        <div key={i} className="surface-plate rounded-2xl p-6">
          <div className="skeleton-sheen h-3 w-24" />
          <div className="skeleton-sheen mt-4 h-5 w-2/3" />
          <div className="skeleton-sheen mt-3 h-3 w-full" />
          <div className="skeleton-sheen mt-2 h-3 w-5/6" />
          <div className="skeleton-sheen mt-2 h-3 w-1/3" />
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
        Nothing in the corpus matches "{query}"
      </h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-muted-foreground">
        Try naming the statute, the doctrine, or the situation in a full sentence —
        Judicia matches on meaning, so more context usually helps.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        <button
          onClick={onReset}
          className="rounded-full border border-border px-4 py-2 text-sm text-ink transition-colors hover:border-accent"
        >
          Start over
        </button>
      </div>
    </div>
  );
}

function Results({ query, results }: { query: string; results: LegalResult[] }) {
  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border pb-4">
        <h2 className="font-display text-lg font-semibold text-ink">
          {results.length} passages retrieved
        </h2>
        <p className="font-mono text-xs text-muted-foreground">
          for "{query.length > 54 ? `${query.slice(0, 54)}…` : query}"
        </p>
      </div>

      <div className="mt-5 space-y-4">
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
        <span className="rounded-md bg-secondary px-2 py-1 text-ink-soft">{result.category}</span>
        <span>{result.jurisdiction}</span>
        <span aria-hidden="true">·</span>
        <span>{result.year}</span>
        <span className="ml-auto flex items-center gap-2 normal-case tracking-normal">
          <span className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
            <span
              className="block h-full rounded-full"
              style={{ width: `${pct}%`, background: "var(--gradient-brass)" }}
            />
          </span>
          {pct}% match
        </span>
      </div>

      <h3 className="mt-4 font-display text-xl font-semibold leading-snug text-ink">
        {result.title}
      </h3>
      <p className="mt-1 text-sm text-muted-foreground">{result.citation}</p>

      <blockquote className="mt-4 border-l-2 border-accent/60 pl-4 text-[15px] leading-relaxed text-ink-soft">
        {result.snippet}
      </blockquote>

      <div className="mt-4 flex flex-wrap gap-1.5">
        {result.passages.map((p) => (
          <span key={p} className="rounded-full bg-secondary px-2.5 py-1 text-xs text-ink-soft">
            {p}
          </span>
        ))}
      </div>

      <div className="mt-5 flex flex-wrap items-end justify-between gap-4 border-t border-border pt-4">
        <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
          <span className="font-semibold text-ink">Why this matched — </span>
          {result.why}
        </p>
        <button className="shrink-0 text-sm font-semibold text-ink underline-offset-4 transition-colors hover:text-accent hover:underline">
          Open document →
        </button>
      </div>
    </article>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-2 px-5 py-8 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p>Judicia · Retrieval-grounded legal research</p>
        <p>Results are research aids, not legal advice.</p>
      </div>
    </footer>
  );
}

export default App;