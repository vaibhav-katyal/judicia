import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowRightIcon,
  BoltIcon,
  BookIcon,
  CheckIcon,
  ChevronDownIcon,
  ClockIcon,
  CopyIcon,
  FileIcon,
  InfoIcon,
  LinkIcon,
  MapPinIcon,
  PhoneIcon,
  RefreshIcon,
  ScaleIcon,
  ShieldIcon,
  SparkIcon,
} from "@/components/Icons";
import type {
  AskResponse,
  FirstAidAnswer,
  HelpEntry,
  LegalSource,
  Urgency,
} from "@/lib/judicia-data";

const URGENCY_META: Record<
  Urgency,
  { label: string; hint: string; className: string; dot: string }
> = {
  emergency: {
    label: "Emergency",
    hint: "If there is immediate danger, call 112 now.",
    className: "border-red-300/80 bg-red-50 text-red-900",
    dot: "bg-red-500",
  },
  urgent: {
    label: "Urgent",
    hint: "Act today or within the next few days.",
    className: "border-amber-300/80 bg-amber-50 text-amber-900",
    dot: "bg-amber-500",
  },
  normal: {
    label: "Standard",
    hint: "Act carefully and keep a written record.",
    className: "border-emerald-300/80 bg-emerald-50 text-emerald-900",
    dot: "bg-emerald-500",
  },
};

const DOMAIN_NOTES: Record<string, string> = {
  Civil: "Contracts, money recovery, and civil disputes.",
  Constitutional: "Fundamental rights and action against public authorities.",
  Consumer: "Defective goods, bad services, refunds, and e-commerce.",
  Criminal: "Threats, assault, theft, trespass, and police complaints.",
  Cyber: "Hacked accounts, online fraud, privacy, and electronic evidence.",
  Family: "Divorce, maintenance, custody, and domestic safety.",
  Labour: "Unpaid salary, termination, maternity, gratuity, and PF.",
  Property: "Tenancy, security deposits, land, title, and partition.",
};

type ResultsViewProps = {
  response: AskResponse;
  onNewQuestion: () => void;
};

export function ResultsView({ response, onNewQuestion }: ResultsViewProps) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const { answer } = response;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(buildPlanText(response));
      setCopied(true);
      setCopyFailed(false);
      window.setTimeout(() => setCopied(false), 2200);
    } catch {
      setCopyFailed(true);
      window.setTimeout(() => setCopyFailed(false), 3200);
    }
  }

  return (
    <div className="space-y-7">
      <div className="no-print flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent">
            Your question
          </p>
          <h1 className="text-balance-tight mt-2 max-w-3xl font-display text-2xl font-semibold leading-snug text-ink sm:text-3xl">
            “{response.query}”
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Analysed in {formatLatency(response.meta.latencyMs)} · {response.sources.length}{" "}
            legal source{response.sources.length === 1 ? "" : "s"} matched
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex h-9 items-center gap-2 rounded-full border border-border bg-card px-3.5 text-xs font-medium text-ink transition-colors hover:border-accent"
          >
            <CopyIcon className="h-4 w-4" />
            {copied ? "Copied" : copyFailed ? "Copy failed" : "Copy plan"}
          </button>
          <button
            type="button"
            onClick={() => window.print()}
            className="inline-flex h-9 items-center gap-2 rounded-full border border-border bg-card px-3.5 text-xs font-medium text-ink transition-colors hover:border-accent"
          >
            <FileIcon className="h-4 w-4" />
            Print
          </button>
          <button
            type="button"
            onClick={onNewQuestion}
            className="ink-panel inline-flex h-9 items-center gap-2 rounded-full px-4 text-xs font-semibold transition-all hover:brightness-125"
          >
            <RefreshIcon className="h-4 w-4" />
            New question
          </button>
        </div>
      </div>

      <AnswerHero response={response} />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.65fr)_minmax(290px,1fr)]">
        <div className="space-y-6">
          <StepsCard answer={answer} />
          <div className="grid gap-6 md:grid-cols-2">
            <RightsCard answer={answer} />
            <OptionsCard answer={answer} />
          </div>
          <HelpCard entries={answer.whereToComplain} />
          <DeadlinesCard deadlines={answer.keyDeadlines} />
          <div className="flex gap-3 rounded-2xl border border-border/80 bg-secondary/40 p-4 text-xs leading-relaxed text-muted-foreground">
            <InfoIcon className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{answer.disclaimer}</p>
          </div>
        </div>

        <aside className="space-y-6 self-start lg:sticky lg:top-24">
          <SnapshotCard response={response} />
          <DocumentsCard answer={answer} />
        </aside>
      </div>

      <SourcesSection
        sources={response.sources}
        engine={response.meta.retrieval.engine}
        reason={response.meta.retrieval.reason}
      />

      <div className="no-print surface-plate rounded-2xl px-6 py-8 text-center">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent">
          Still need help?
        </p>
        <h3 className="mt-2 font-display text-xl font-semibold text-ink">
          Ask about a different problem
        </h3>
        <p className="mx-auto mt-2 max-w-lg text-sm leading-relaxed text-muted-foreground">
          Describe another situation and Judicia will classify it, pull the closest
          statutory provisions, and build a fresh first-aid plan.
        </p>
        <button
          type="button"
          onClick={onNewQuestion}
          className="ink-panel mt-5 inline-flex h-11 items-center gap-2 rounded-full px-6 text-sm font-semibold transition-all hover:brightness-125"
        >
          Ask another question
          <ArrowRightIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function AnswerHero({ response }: { response: AskResponse }) {
  const urgency = URGENCY_META[response.answer.urgency];
  const confidence = Math.round(response.confidence * 100);
  const retrievalLabel =
    response.meta.retrieval.engine === "pinecone" ? "Pinecone" : "Local index";
  const draftLabel = response.meta.llm.used
    ? response.meta.llm.model
    : "Rule-based plan";

  return (
    <section className="relative overflow-hidden rounded-2xl border border-accent/35 bg-card p-6 shadow-[0_18px_50px_-30px_color-mix(in_oklab,var(--ink)_45%,transparent)] sm:p-8">
      <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-accent/10 blur-3xl" />
      <div className="relative">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${urgency.className}`}
          >
            <span className={`h-2 w-2 rounded-full ${urgency.dot}`} />
            {urgency.label}
          </span>
          <span className="rounded-full border border-border bg-secondary/70 px-3 py-1 text-xs font-medium text-ink-soft">
            {response.domain} law
          </span>
          <span className="rounded-full border border-border bg-card px-3 py-1 font-mono text-[11px] text-muted-foreground">
            {confidence}% classifier confidence
          </span>
        </div>

        <h2 className="mt-4 font-display text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Your first-aid plan
        </h2>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-ink-soft sm:text-base">
          {response.answer.summary}
        </p>

        <p className="mt-3 text-xs font-medium text-muted-foreground">{urgency.hint}</p>

        {response.answer.urgency === "emergency" && (
          <a
            href="tel:112"
            className="mt-4 inline-flex items-center gap-2 rounded-xl border border-red-300 bg-red-50 px-4 py-2.5 text-sm font-semibold text-red-900 transition-colors hover:bg-red-100"
          >
            <PhoneIcon className="h-4 w-4" />
            Call 112 now if you are in danger
          </a>
        )}

        <div className="mt-6 grid gap-2 sm:grid-cols-3">
          <PipelineChip
            icon={<ScaleIcon className="h-4 w-4" />}
            label="Classified"
            value={`${response.domain} · ${confidence}%`}
          />
          <PipelineChip
            icon={<BookIcon className="h-4 w-4" />}
            label="Retrieved"
            value={`${response.sources.length} sources · ${retrievalLabel}`}
          />
          <PipelineChip
            icon={<SparkIcon className="h-4 w-4" />}
            label="Drafted"
            value={draftLabel}
          />
        </div>
      </div>
    </section>
  );
}

function PipelineChip({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border/80 bg-secondary/35 px-3 py-2.5">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-card text-accent shadow-sm">
        {icon}
      </span>
      <span className="min-w-0">
        <span className="block font-mono text-[9px] uppercase tracking-[0.16em] text-muted-foreground">
          {label}
        </span>
        <span className="block truncate text-xs font-semibold text-ink" title={value}>
          {value}
        </span>
      </span>
    </div>
  );
}

function Panel({
  eyebrow,
  title,
  icon,
  children,
  className = "",
}: {
  eyebrow: string;
  title: string;
  icon: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`surface-plate rounded-2xl p-5 sm:p-6 ${className}`}>
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary text-accent">
          {icon}
        </span>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            {eyebrow}
          </p>
          <h3 className="mt-0.5 font-display text-lg font-semibold text-ink">{title}</h3>
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function StepsCard({ answer }: { answer: FirstAidAnswer }) {
  return (
    <Panel
      eyebrow="Start here"
      title="Do this now"
      icon={<BoltIcon className="h-5 w-5" />}
    >
      {answer.immediateSteps.length === 0 ? (
        <p className="text-sm leading-relaxed text-muted-foreground">
          No specific steps were generated. Preserve your evidence and contact the District
          Legal Services Authority for guidance.
        </p>
      ) : (
        <ol className="space-y-4">
          {answer.immediateSteps.map((step, index) => (
            <li key={`${index}-${step.slice(0, 32)}`} className="flex gap-3.5">
              <span className="ink-panel mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold">
                {index + 1}
              </span>
              <p className="pt-0.5 text-[15px] leading-relaxed text-ink-soft">{step}</p>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

function RightsCard({ answer }: { answer: FirstAidAnswer }) {
  return (
    <Panel
      eyebrow="Legal basis"
      title="Your rights & the law"
      icon={<ShieldIcon className="h-5 w-5" />}
    >
      {answer.rights.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          See the statutory sources below for the provisions that apply.
        </p>
      ) : (
        <ul className="space-y-3">
          {answer.rights.map((right, index) => (
            <li key={`${index}-${right.slice(0, 28)}`} className="flex gap-2.5">
              <CheckIcon className="mt-1 h-4 w-4 shrink-0 text-accent" />
              <p className="text-sm leading-relaxed text-ink-soft">{right}</p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function OptionsCard({ answer }: { answer: FirstAidAnswer }) {
  return (
    <Panel
      eyebrow="Routes available"
      title="Other options"
      icon={<ScaleIcon className="h-5 w-5" />}
    >
      {answer.options.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Ask a lawyer about the best route for your facts.
        </p>
      ) : (
        <ul className="space-y-3">
          {answer.options.map((option, index) => (
            <li key={`${index}-${option.slice(0, 28)}`} className="flex gap-2.5">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
              <p className="text-sm leading-relaxed text-ink-soft">{option}</p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function HelpCard({ entries }: { entries: HelpEntry[] }) {
  return (
    <Panel
      eyebrow="Where to go"
      title="Get help or file a complaint"
      icon={<MapPinIcon className="h-5 w-5" />}
    >
      {entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Contact your District Legal Services Authority for the correct forum.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {entries.map((entry, index) => (
            <li
              key={`${index}-${entry.name}`}
              className="flex flex-col gap-3 rounded-xl border border-border/80 bg-background/60 p-3.5 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0">
                <p className="text-sm font-semibold text-ink">{entry.name}</p>
                {entry.detail && (
                  <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                    {entry.detail}
                  </p>
                )}
              </div>
              <div className="flex shrink-0 flex-wrap gap-2">
                {entry.phone && (
                  <a
                    href={phoneHref(entry.phone)}
                    className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border bg-card px-3 text-xs font-semibold text-ink transition-colors hover:border-accent"
                  >
                    <PhoneIcon className="h-3.5 w-3.5" />
                    {entry.phone}
                  </a>
                )}
                {entry.link && isSafeLink(entry.link) && (
                  <a
                    href={entry.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border bg-card px-3 text-xs font-semibold text-ink transition-colors hover:border-accent"
                  >
                    <LinkIcon className="h-3.5 w-3.5" />
                    Open
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function DeadlinesCard({ deadlines }: { deadlines: string[] }) {
  if (deadlines.length === 0) return null;

  return (
    <section className="rounded-2xl border border-amber-300/70 bg-amber-50/70 p-5 sm:p-6">
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-100 text-amber-800">
          <ClockIcon className="h-5 w-5" />
        </span>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-amber-700">
            Do not miss
          </p>
          <h3 className="mt-0.5 font-display text-lg font-semibold text-amber-950">
            Time-sensitive points
          </h3>
        </div>
      </div>
      <ul className="mt-4 space-y-2.5">
        {deadlines.map((deadline, index) => (
          <li key={`${index}-${deadline.slice(0, 28)}`} className="flex gap-2.5">
            <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />
            <p className="text-sm leading-relaxed text-amber-950/85">{deadline}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

function SnapshotCard({ response }: { response: AskResponse }) {
  const sortedProbabilities = useMemo(
    () =>
      Object.entries(response.probabilities)
        .map(([label, score]) => ({ label, score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5),
    [response.probabilities]
  );

  const confidence = Math.round(response.confidence * 100);
  const retrieval = response.meta.retrieval;
  const llm = response.meta.llm;

  return (
    <section className="surface-plate rounded-2xl p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            Case snapshot
          </p>
          <h3 className="mt-1 font-display text-xl font-semibold text-ink">
            {response.domain}
          </h3>
        </div>
        <span className="rounded-full bg-secondary px-2.5 py-1 font-mono text-xs font-semibold text-ink-soft">
          {confidence}%
        </span>
      </div>

      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
        {DOMAIN_NOTES[response.domain] ?? "A legal issue identified from your description."}
      </p>

      {sortedProbabilities.length > 0 && (
        <div className="mt-5">
          <div className="flex items-center justify-between">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
              Classifier distribution
            </p>
            <span className="text-[10px] text-muted-foreground">top 5</span>
          </div>
          <div className="mt-2.5 space-y-2">
            {sortedProbabilities.map((entry) => {
              const percentage = Math.round(entry.score * 100);
              const isTop = entry.label === response.domain;
              return (
                <div key={entry.label}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className={isTop ? "font-semibold text-ink" : "text-muted-foreground"}>
                      {entry.label}
                    </span>
                    <span className={isTop ? "font-mono font-semibold text-accent" : "font-mono text-muted-foreground"}>
                      {percentage}%
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${Math.max(2, percentage)}%`,
                        backgroundColor: isTop ? "var(--accent)" : "color-mix(in oklab, var(--ink) 22%, transparent)",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <dl className="mt-5 space-y-2 border-t border-border pt-4 text-[11px]">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">Retrieval</dt>
          <dd className="font-medium text-ink">
            {retrieval.engine === "pinecone" ? "Pinecone" : "Local keyword index"}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">Drafting</dt>
          <dd className="max-w-[65%] truncate text-right font-medium text-ink" title={llm.model}>
            {llm.used ? llm.model : "Rule-based playbook"}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">Knowledge base</dt>
          <dd className="font-medium text-ink">{response.meta.corpusItems || "—"} provisions</dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">Total time</dt>
          <dd className="font-medium text-ink">{formatLatency(response.meta.latencyMs)}</dd>
        </div>
      </dl>

      {!llm.used && llm.reason && (
        <details className="mt-4 rounded-lg border border-amber-200 bg-amber-50/60 px-3 py-2">
          <summary className="cursor-pointer text-[11px] font-semibold text-amber-900">
            Why is this a rule-based plan?
          </summary>
          <p className="mt-1.5 text-[11px] leading-relaxed text-amber-900/80">{llm.reason}</p>
        </details>
      )}
    </section>
  );
}

function DocumentsCard({ answer }: { answer: FirstAidAnswer }) {
  const [checked, setChecked] = useState<Record<number, boolean>>({});

  return (
    <Panel
      eyebrow="Build your file"
      title="Documents to gather"
      icon={<FileIcon className="h-5 w-5" />}
    >
      {answer.documents.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Collect every notice, receipt, message, and photograph connected to the issue.
        </p>
      ) : (
        <>
          <p className="-mt-1 mb-3 text-xs text-muted-foreground">
            Tap to tick off what you already have.
          </p>
          <ul className="space-y-2.5">
            {answer.documents.map((document, index) => {
              const isChecked = Boolean(checked[index]);
              return (
                <li key={`${index}-${document.slice(0, 28)}`}>
                  <button
                    type="button"
                    aria-pressed={isChecked}
                    onClick={() =>
                      setChecked((current) => ({ ...current, [index]: !current[index] }))
                    }
                    className="flex w-full items-start gap-2.5 text-left"
                  >
                    <span
                      className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${
                        isChecked
                          ? "border-accent bg-accent text-accent-foreground"
                          : "border-border bg-background"
                      }`}
                    >
                      {isChecked && <CheckIcon className="h-3 w-3" />}
                    </span>
                    <span
                      className={`text-sm leading-relaxed transition-colors ${
                        isChecked ? "text-muted-foreground line-through" : "text-ink-soft"
                      }`}
                    >
                      {document}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </Panel>
  );
}

function SourcesSection({
  sources,
  engine,
  reason,
}: {
  sources: LegalSource[];
  engine: string;
  reason?: string;
}) {
  return (
    <section id="sources" className="scroll-mt-24">
      <div className="flex flex-col gap-2 border-b border-border pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-accent">
            Grounded in law
          </p>
          <h2 className="mt-1 font-display text-xl font-semibold text-ink">
            The sources behind this answer
          </h2>
        </div>
        <div className="text-left sm:text-right">
          <p className="text-xs text-muted-foreground">
            {sources.length} provisions ·{" "}
            {engine === "pinecone" ? "Pinecone semantic search" : "local keyword search"}
          </p>
          {engine !== "pinecone" && reason && (
            <p className="mt-0.5 max-w-md text-[11px] text-amber-700">{reason}</p>
          )}
        </div>
      </div>

      {sources.length === 0 ? (
        <div className="surface-plate mt-4 rounded-2xl p-6 text-sm text-muted-foreground">
          No statutory source was retrieved for this question. Use the steps above and
          speak to a lawyer before acting. You can also try rephrasing with the people,
          dates, and documents involved.
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {sources.map((source, index) => (
            <SourceCard
              key={source.chunkId || source.id || index}
              source={source}
              index={index}
              defaultOpen={index === 0}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function SourceCard({
  source,
  index,
  defaultOpen,
}: {
  source: LegalSource;
  index: number;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const match = Math.round(source.match * 100);

  return (
    <article className="surface-plate overflow-hidden rounded-2xl">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-start gap-3.5 p-4 text-left transition-colors hover:bg-secondary/25 sm:p-5"
      >
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-secondary font-mono text-xs font-bold text-ink-soft">
          {String(index + 1).padStart(2, "0")}
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-2.5 gap-y-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
            <span className="rounded bg-secondary px-1.5 py-0.5 font-semibold text-ink-soft">
              {source.category}
            </span>
            <span>{source.domain}</span>
            {source.year > 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span>{source.year}</span>
              </>
            )}
            <span className="ml-auto text-[10px] normal-case tracking-normal text-accent">
              {match}% match
            </span>
          </span>

          <span className="mt-2 block font-display text-base font-semibold leading-snug text-ink sm:text-lg">
            {source.title}
          </span>
          {source.citation && (
            <span className="mt-1 block font-mono text-[11px] text-muted-foreground">
              {source.citation} · {source.jurisdiction}
            </span>
          )}
        </span>

        <ChevronDownIcon
          className={`mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open && (
        <div className="border-t border-border px-4 pb-5 pt-4 sm:px-5">
          {source.snippet && (
            <blockquote className="rounded-r-lg border-l-2 border-accent bg-secondary/30 py-2 pl-4 pr-3 text-sm leading-relaxed text-ink-soft">
              {source.snippet}
            </blockquote>
          )}

          {source.passages.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {source.passages.map((passage, passageIndex) => (
                <span
                  key={`${passageIndex}-${passage.slice(0, 20)}`}
                  className="rounded-full border border-border bg-card px-2.5 py-1 font-mono text-[10px] text-ink-soft"
                >
                  {passage}
                </span>
              ))}
            </div>
          )}

          {source.why && (
            <p className="mt-3 flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground">
              <InfoIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              {source.why}
            </p>
          )}
        </div>
      )}
    </article>
  );
}

function phoneHref(phone: string): string {
  return `tel:${phone.replace(/[^\d+]/g, "")}`;
}

function isSafeLink(link: string): boolean {
  try {
    const url = new URL(link);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch {
    return false;
  }
}

function formatLatency(milliseconds: number | undefined): string {
  if (!milliseconds || milliseconds <= 0) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function buildPlanText(response: AskResponse): string {
  const { answer } = response;
  const lines: string[] = [
    "JUDICIA — LEGAL FIRST-AID PLAN",
    "",
    `Question: ${response.query}`,
    `Category: ${response.domain} (${Math.round(response.confidence * 100)}% classifier confidence)`,
    `Urgency: ${answer.urgency.toUpperCase()}`,
    "",
    "SUMMARY",
    answer.summary,
    "",
    "DO THIS NOW",
    ...numbered(answer.immediateSteps),
    "",
    "YOUR RIGHTS & THE LAW",
    ...bulleted(answer.rights),
    "",
    "OTHER OPTIONS",
    ...bulleted(answer.options),
    "",
    "DOCUMENTS TO GATHER",
    ...bulleted(answer.documents),
    "",
    "WHERE TO GET HELP",
    ...answer.whereToComplain.map((entry) => {
      const contact = [entry.phone, entry.link].filter(Boolean).join(" · ");
      return `- ${entry.name}${entry.detail ? ` — ${entry.detail}` : ""}${
        contact ? ` (${contact})` : ""
      }`;
    }),
  ];

  if (answer.keyDeadlines.length > 0) {
    lines.push("", "TIME-SENSITIVE POINTS", ...bulleted(answer.keyDeadlines));
  }

  if (response.sources.length > 0) {
    lines.push(
      "",
      "SOURCES",
      ...response.sources.map((source, index) => {
        const citation = source.citation || source.title;
        return `[S${index + 1}] ${citation} — ${source.title}`;
      })
    );
  }

  lines.push("", "IMPORTANT", answer.disclaimer);
  return lines.join("\n");
}

function numbered(items: string[]): string[] {
  return items.map((item, index) => `${index + 1}. ${item}`);
}

function bulleted(items: string[]): string[] {
  return items.length ? items.map((item) => `- ${item}`) : ["- None listed"];
}
