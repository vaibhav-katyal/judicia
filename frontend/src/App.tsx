import { useCallback, useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Composer } from "@/components/Composer";
import { ResultsView } from "@/components/ResultsView";
import {
  AlertIcon,
  ArrowRightIcon,
  BookIcon,
  CheckIcon,
  PhoneIcon,
  RefreshIcon,
  ScaleIcon,
  SparkIcon,
} from "@/components/Icons";
import {
  askJudicia,
  checkHealth,
  DOMAINS,
  EXAMPLE_QUERIES,
  type AskResponse,
  type HealthResponse,
} from "@/lib/judicia-data";

type Phase = "idle" | "loading" | "result" | "error";
type HealthTone = "ready" | "partial" | "local" | "offline";

const RECENT_KEY = "judicia:recent-questions:v1";

function App() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [recent, setRecent] = useState<string[]>(readRecent);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const askController = useRef<AbortController | null>(null);
  const healthController = useRef<AbortController | null>(null);

  const refreshHealth = useCallback(async () => {
    healthController.current?.abort();
    const controller = new AbortController();
    healthController.current = controller;
    const result = await checkHealth(controller.signal);
    if (!controller.signal.aborted) setHealth(result);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshHealth();
    }, 0);
    return () => {
      window.clearTimeout(timer);
      healthController.current?.abort();
      askController.current?.abort();
    };
  }, [refreshHealth]);

  useEffect(() => {
    if (phase !== "idle") return;
    const timer = window.setTimeout(() => textareaRef.current?.focus(), 80);
    return () => window.clearTimeout(timer);
  }, [phase]);

  const handleAsk = useCallback(async (raw: string) => {
    const cleaned = raw.trim();
    if (!cleaned) return;

    askController.current?.abort();
    const controller = new AbortController();
    askController.current = controller;

    setQuery(cleaned);
    setSubmitted(cleaned);
    setPhase("loading");
    setError(null);
    setResponse(null);

    try {
      const data = await askJudicia(cleaned, controller.signal);
      if (controller.signal.aborted) return;
      setResponse(data);
      setPhase("result");
      setRecent((current) => {
        const next = [cleaned, ...current.filter((item) => item !== cleaned)].slice(0, 4);
        writeRecent(next);
        return next;
      });
      window.scrollTo({ top: 0, behavior: "smooth" });
      void refreshHealth();
    } catch (caught) {
      if (controller.signal.aborted) return;
      if (caught instanceof Error && caught.name === "AbortError") return;
      setError(
        caught instanceof Error
          ? caught.message
          : "Something went wrong while building your plan."
      );
      setPhase("error");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      if (askController.current === controller) askController.current = null;
    }
  }, [refreshHealth]);

  const handleNewQuestion = useCallback(() => {
    askController.current?.abort();
    setQuery("");
    setSubmitted("");
    setResponse(null);
    setError(null);
    setPhase("idle");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header
        health={health}
        onRefreshHealth={refreshHealth}
        onNewQuestion={handleNewQuestion}
        showNewQuestion={phase !== "idle"}
      />

      <main className="mx-auto w-full max-w-6xl px-4 pb-20 sm:px-6 lg:px-8">
        {phase === "idle" && (
          <Landing
            query={query}
            onQueryChange={setQuery}
            onSubmit={() => void handleAsk(query)}
            onExample={(exampleQuery) => void handleAsk(exampleQuery)}
            recent={recent}
            onRecent={(recentQuery) => void handleAsk(recentQuery)}
            textareaRef={textareaRef}
          />
        )}

        {phase === "loading" && <LoadingState query={submitted} />}

        {phase === "error" && (
          <ErrorState
            message={error ?? "The request failed."}
            query={submitted}
            onRetry={() => void handleAsk(submitted)}
            onReset={handleNewQuestion}
          />
        )}

        {phase === "result" && response && (
          <ResultsView response={response} onNewQuestion={handleNewQuestion} />
        )}
      </main>

      <Footer />
    </div>
  );
}

function Header({
  health,
  onRefreshHealth,
  onNewQuestion,
  showNewQuestion,
}: {
  health: HealthResponse | null;
  onRefreshHealth: () => void;
  onNewQuestion: () => void;
  showNewQuestion: boolean;
}) {
  return (
    <header className="no-print sticky top-0 z-30 border-b border-border/70 bg-background/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 px-4 py-3.5 sm:px-6 lg:px-8">
        <button
          type="button"
          onClick={onNewQuestion}
          className="flex min-w-0 items-center gap-2.5 text-left"
          aria-label="Judicia home"
        >
          <span className="ink-panel flex h-9 w-9 shrink-0 items-center justify-center rounded-xl">
            <ScaleIcon className="h-4 w-4 text-accent" />
          </span>
          <span className="min-w-0">
            <span className="flex items-center gap-2">
              <span className="font-display text-lg font-semibold tracking-tight text-ink">
                Judicia
              </span>
              <span className="hidden rounded bg-secondary/80 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-muted-foreground sm:inline">
                v2.0 RAG
              </span>
            </span>
            <span className="hidden text-[10px] text-muted-foreground sm:block">
              Legal first aid, in plain language
            </span>
          </span>
        </button>

        <div className="flex shrink-0 items-center gap-2">
          <StatusPill health={health} onRefresh={onRefreshHealth} />
          {showNewQuestion && (
            <button
              type="button"
              onClick={onNewQuestion}
              className="inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-card px-3 text-xs font-semibold text-ink transition-colors hover:border-accent"
            >
              <RefreshIcon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">New question</span>
              <span className="sm:hidden">New</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

function StatusPill({
  health,
  onRefresh,
}: {
  health: HealthResponse | null;
  onRefresh: () => void;
}) {
  const summary = getHealthSummary(health);

  return (
    <button
      type="button"
      onClick={onRefresh}
      title={`${summary.detail} Click to refresh.`}
      aria-label={`${summary.label}. ${summary.detail} Click to refresh.`}
      className="group inline-flex h-9 items-center gap-2 rounded-full border border-border bg-card px-2.5 text-xs text-muted-foreground transition-colors hover:border-accent sm:px-3"
    >
      <span className="flex items-center gap-1">
        {summary.dots.map((dot) => (
          <span
            key={dot.label}
            title={`${dot.label}: ${dot.ok === null ? "unknown" : dot.ok ? "ready" : "not configured"}`}
            className={`h-1.5 w-1.5 rounded-full ${
              dot.ok === null
                ? "bg-gray-400"
                : dot.ok
                  ? "bg-emerald-500"
                  : "bg-amber-500"
            }`}
          />
        ))}
      </span>
      <span className={`hidden font-medium sm:inline ${summary.labelClass}`}>
        {summary.label}
      </span>
      <span className={`sm:hidden ${summary.labelClass}`}>
        {summary.tone === "ready" ? "AI ready" : summary.tone === "offline" ? "Offline" : "Fallback"}
      </span>
      <RefreshIcon className="h-3 w-3 opacity-45 transition-opacity group-hover:opacity-80" />
    </button>
  );
}

function getHealthSummary(health: HealthResponse | null): {
  tone: HealthTone;
  label: string;
  labelClass: string;
  detail: string;
  dots: { label: string; ok: boolean | null }[];
} {
  if (!health) {
    return {
      tone: "offline",
      label: "Engine offline",
      labelClass: "text-amber-700",
      detail:
        "The backend is not reachable. Start it with: python backend/app.py",
      dots: [
        { label: "Classifier", ok: null },
        { label: "Pinecone", ok: null },
        { label: "Groq", ok: null },
      ],
    };
  }

  const classifierReady = health.classifier.loaded;
  const pineconeReady = health.rag.pinecone.configured;
  const groqReady = health.rag.groq.configured;
  const dots = [
    { label: "Classifier", ok: classifierReady },
    { label: "Pinecone", ok: pineconeReady },
    { label: "Groq", ok: groqReady },
  ];

  const detail =
    `Classifier: ${classifierReady ? "ready" : "unavailable"} · ` +
    `Retrieval: ${pineconeReady ? `Pinecone (${health.rag.pinecone.index})` : "local index"} · ` +
    `Drafting: ${groqReady ? health.rag.groq.model : "rule-based playbook"}`;

  if (classifierReady && pineconeReady && groqReady) {
    return {
      tone: "ready",
      label: "AI engine ready",
      labelClass: "text-emerald-700",
      detail,
      dots,
    };
  }

  if (classifierReady && !pineconeReady && !groqReady) {
    return {
      tone: "local",
      label: "Local fallback mode",
      labelClass: "text-amber-700",
      detail:
        `${detail}. Add Pinecone and Groq keys to backend/.env for full RAG generation.`,
      dots,
    };
  }

  return {
    tone: "partial",
    label: "Limited AI mode",
    labelClass: "text-amber-700",
    detail: `${detail}. Check backend/.env to enable all services.`,
    dots,
  };
}

function Landing({
  query,
  onQueryChange,
  onSubmit,
  onExample,
  recent,
  onRecent,
  textareaRef,
}: {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  onExample: (query: string) => void;
  recent: string[];
  onRecent: (query: string) => void;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}) {
  return (
    <div className="space-y-14">
      <section className="rule-grid hero-glow relative -mx-4 overflow-hidden px-4 pb-10 pt-12 sm:-mx-6 sm:px-6 sm:pb-14 sm:pt-16 lg:-mx-8 lg:px-8 lg:pt-20">
        <div className="mx-auto max-w-4xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card/85 px-3.5 py-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground shadow-sm">
            <SparkIcon className="h-3.5 w-3.5 text-accent" />
            India-specific legal first aid
          </span>

          <h1 className="text-balance-tight mt-6 font-display text-5xl font-semibold tracking-tight text-ink sm:text-6xl lg:text-7xl">
            Legal trouble?
            <span className="mt-1 block text-brass-gradient">Know your next move.</span>
          </h1>

          <p className="mx-auto mt-5 max-w-2xl text-balance-tight text-base leading-relaxed text-muted-foreground sm:text-lg">
            Describe what happened in your own words. Judicia identifies the area of law,
            retrieves the governing provisions, and gives you a step-by-step first-aid plan.
          </p>

          <div className="mx-auto mt-8 max-w-3xl text-left">
            <Composer
              value={query}
              onChange={onQueryChange}
              onSubmit={onSubmit}
              autoFocus
              textareaRef={textareaRef}
            />
          </div>

          <p className="mt-4 text-xs text-muted-foreground">
            <span className="font-semibold text-red-700">In immediate danger?</span>{" "}
            Call{" "}
            <a
              href="tel:112"
              className="inline-flex items-center gap-1 font-semibold text-red-700 underline decoration-red-300 underline-offset-2 hover:text-red-800"
            >
              <PhoneIcon className="h-3 w-3" />
              112
            </a>{" "}
            first. Judicia is legal information, not emergency response.
          </p>

          <div className="mt-9">
            <div className="flex items-center justify-center gap-3">
              <span className="h-px w-8 bg-border" />
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                Try a situation
              </span>
              <span className="h-px w-8 bg-border" />
            </div>

            <div className="mt-4 grid gap-2.5 text-left sm:grid-cols-2 lg:grid-cols-3">
              {EXAMPLE_QUERIES.map((example) => (
                <button
                  key={example.label}
                  type="button"
                  onClick={() => onExample(example.query)}
                  className="surface-plate-interactive group rounded-xl p-4 text-left"
                >
                  <span className="flex items-center justify-between gap-3">
                    <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-accent">
                      {example.label}
                    </span>
                    <ArrowRightIcon className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
                  </span>
                  <span className="mt-1.5 block text-sm leading-snug text-ink-soft">
                    {example.query}
                  </span>
                </button>
              ))}
            </div>

            {recent.length > 0 && (
              <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                  Recent
                </span>
                {recent.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => onRecent(item)}
                    className="max-w-[280px] truncate rounded-full border border-border bg-card/70 px-3 py-1 text-[11px] text-muted-foreground transition-colors hover:border-accent hover:text-ink"
                    title={item}
                  >
                    {item}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="pt-2">
        <div className="mx-auto max-w-3xl text-center">
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-accent">
            How it works
          </p>
          <h2 className="mt-2 font-display text-2xl font-semibold text-ink sm:text-3xl">
            From “what do I do?” to a plan
          </h2>
          <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-muted-foreground">
            Three transparent steps. You can see exactly which sources and models were
            used in every answer.
          </p>
        </div>

        <div className="mt-7 grid gap-3 md:grid-cols-3">
          <HowItWorksCard
            number="01"
            icon={<ScaleIcon className="h-5 w-5" />}
            title="Reads your situation"
            body="The fine-tuned InLegalBERT classifier maps your question to one of eight legal domains and shows its confidence."
          />
          <HowItWorksCard
            number="02"
            icon={<BookIcon className="h-5 w-5" />}
            title="Finds the governing law"
            body="Pinecone semantic search retrieves the closest statutes and precedents. If Pinecone is unavailable, a local index takes over."
          />
          <HowItWorksCard
            number="03"
            icon={<SparkIcon className="h-5 w-5" />}
            title="Drafts your first-aid plan"
            body="Groq turns the retrieved provisions into concrete steps, your rights, documents to gather, and where to complain — grounded in the sources."
          />
        </div>
      </section>

      <section className="surface-plate rounded-2xl p-6 sm:p-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-accent">
              Coverage
            </p>
            <h2 className="mt-1 font-display text-xl font-semibold text-ink sm:text-2xl">
              Eight areas of Indian law
            </h2>
          </div>
          <p className="max-w-2xl text-xs leading-relaxed text-muted-foreground lg:text-right">
            The curated knowledge base includes BNS 2023, BSA 2023, IPC, the IT Act 2000,
            the Consumer Protection Act 2019, the Constitution of India, property and
            tenancy statutes, labour law, and family law.
          </p>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {DOMAINS.map((domain) => (
            <span
              key={domain}
              className="rounded-full border border-border bg-secondary/50 px-3.5 py-1.5 text-xs font-medium text-ink-soft"
            >
              {domain}
            </span>
          ))}
        </div>

        <div className="mt-6 grid gap-3 border-t border-border pt-5 text-xs text-muted-foreground sm:grid-cols-3">
          <span className="flex items-center gap-2">
            <CheckIcon className="h-3.5 w-3.5 text-emerald-600" />
            Source citations on every answer
          </span>
          <span className="flex items-center gap-2">
            <CheckIcon className="h-3.5 w-3.5 text-emerald-600" />
            Plain-language, action-first steps
          </span>
          <span className="flex items-center gap-2">
            <CheckIcon className="h-3.5 w-3.5 text-emerald-600" />
            Local fallback if an API is down
          </span>
        </div>
      </section>
    </div>
  );
}

function HowItWorksCard({
  number,
  icon,
  title,
  body,
}: {
  number: string;
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <article className="surface-plate rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-secondary text-accent">
          {icon}
        </span>
        <span className="font-mono text-xs tracking-[0.2em] text-muted-foreground/70">
          {number}
        </span>
      </div>
      <h3 className="mt-4 font-display text-lg font-semibold text-ink">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{body}</p>
    </article>
  );
}

function LoadingState({ query }: { query: string }) {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const second = window.setTimeout(() => setStage(1), 1600);
    const third = window.setTimeout(() => setStage(2), 4200);
    return () => {
      window.clearTimeout(second);
      window.clearTimeout(third);
    };
  }, []);

  const stages = [
    {
      title: "Reading your situation",
      detail: "The InLegalBERT classifier is identifying the area of law.",
    },
    {
      title: "Finding the governing provisions",
      detail: "Searching the legal knowledge base for the closest statutes and precedents.",
    },
    {
      title: "Drafting your first-aid plan",
      detail: "Turning the retrieved law into clear, grounded next steps.",
    },
  ];

  return (
    <div className="mx-auto max-w-3xl py-10 sm:py-14" role="status" aria-live="polite">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-accent">
        Working on it
      </p>
      <h1 className="mt-2 font-display text-3xl font-semibold text-ink sm:text-4xl">
        Building your first-aid plan
      </h1>
      <p className="mt-2 line-clamp-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        “{query}”
      </p>

      <div className="surface-plate mt-8 rounded-2xl p-5 sm:p-7">
        <div className="space-y-1">
          {stages.map((item, index) => {
            const done = index < stage;
            const active = index === stage;
            return (
              <div
                key={item.title}
                className={`flex items-start gap-3.5 rounded-xl px-2 py-3 transition-colors ${
                  active ? "bg-secondary/45" : ""
                }`}
              >
                <span
                  className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs font-bold ${
                    done
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : active
                        ? "ink-panel border-transparent"
                        : "border-border bg-card text-muted-foreground"
                  }`}
                >
                  {done ? <CheckIcon className="h-3.5 w-3.5" /> : index + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p
                    className={`text-sm font-semibold transition-colors ${
                      done || active ? "text-ink" : "text-muted-foreground"
                    }`}
                  >
                    {item.title}
                  </p>
                  <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                    {item.detail}
                  </p>
                </div>
                {active && (
                  <span className="mt-2 h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                )}
                {done && (
                  <span className="mt-1.5 shrink-0 font-mono text-[9px] uppercase tracking-wide text-emerald-700">
                    done
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-5 space-y-3 border-t border-border pt-5">
          <div className="skeleton-sheen h-3 w-40 rounded" />
          <div className="skeleton-sheen h-5 w-3/4 rounded" />
          <div className="skeleton-sheen h-3 w-full rounded" />
          <div className="skeleton-sheen h-3 w-5/6 rounded" />
        </div>
      </div>
    </div>
  );
}

function ErrorState({
  message,
  query,
  onRetry,
  onReset,
}: {
  message: string;
  query: string;
  onRetry: () => void;
  onReset: () => void;
}) {
  return (
    <div className="mx-auto max-w-2xl py-10 sm:py-14">
      <div className="surface-plate rounded-2xl p-6 text-center sm:p-9">
        <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-amber-50 text-amber-700">
          <AlertIcon className="h-5 w-5" />
        </span>
        <p className="mt-5 font-mono text-[10px] uppercase tracking-[0.18em] text-amber-700">
          Could not finish the plan
        </p>
        <h1 className="mt-2 font-display text-2xl font-semibold text-ink sm:text-3xl">
          The Judicia engine did not respond
        </h1>
        {query && (
          <p className="mx-auto mt-2 max-w-lg truncate text-xs text-muted-foreground" title={query}>
            “{query}”
          </p>
        )}
        <p className="mx-auto mt-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
          {message}
        </p>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={onRetry}
            className="ink-panel inline-flex h-10 items-center gap-2 rounded-full px-5 text-sm font-semibold transition-all hover:brightness-125"
          >
            <RefreshIcon className="h-4 w-4" />
            Try again
          </button>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-border bg-card px-5 text-sm font-semibold text-ink transition-colors hover:border-accent"
          >
            Start over
            <ArrowRightIcon className="h-4 w-4" />
          </button>
        </div>

        <details className="mx-auto mt-7 max-w-md rounded-xl border border-border bg-secondary/30 px-4 py-3 text-left">
          <summary className="cursor-pointer text-xs font-semibold text-ink">
            Backend quick check
          </summary>
          <ol className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
            <li>
              1. Start the API:{" "}
              <code className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px] text-ink">
                python backend/app.py
              </code>
            </li>
            <li>
              2. Check it loads at{" "}
              <code className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px] text-ink">
                http://localhost:8000/api/health
              </code>
            </li>
            <li>
              3. For full RAG, copy{" "}
              <code className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px] text-ink">
                backend/.env.example
              </code>{" "}
              to{" "}
              <code className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px] text-ink">
                backend/.env
              </code>{" "}
              and add your Pinecone + Groq keys.
            </li>
            <li>
              4. Seed the vector database:{" "}
              <code className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px] text-ink">
                python backend/ingest_pinecone.py
              </code>
            </li>
          </ol>
        </details>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="no-print mt-8 border-t border-border">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-8 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <div>
          <p className="font-display text-sm font-semibold text-ink">Judicia</p>
          <p className="mt-1">
            Legal information and first-aid guidance — not a substitute for a lawyer.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span>InLegalBERT classifier</span>
          <span aria-hidden="true">·</span>
          <span>Pinecone retrieval</span>
          <span aria-hidden="true">·</span>
          <span>Groq drafting</span>
          <a href="tel:112" className="font-semibold text-red-700 hover:text-red-800">
            Emergency: 112
          </a>
        </div>
      </div>
    </footer>
  );
}

function readRecent(): string[] {
  try {
    const raw = window.localStorage.getItem(RECENT_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is string => typeof item === "string").slice(0, 4);
  } catch {
    return [];
  }
}

function writeRecent(items: string[]) {
  try {
    window.localStorage.setItem(RECENT_KEY, JSON.stringify(items));
  } catch {
    // Private mode or storage disabled: recent history is optional.
  }
}

export default App;
