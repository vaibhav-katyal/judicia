/**
 * Judicia frontend API client.
 *
 * The backend returns snake_case JSON. Everything is normalised here so the
 * React components can rely on a single, safe camelCase shape even if the
 * backend is an older build or a field is missing.
 */

export type Urgency = "emergency" | "urgent" | "normal";

export type HelpEntry = {
  name: string;
  detail: string;
  phone: string | null;
  link: string | null;
};

export type FirstAidAnswer = {
  summary: string;
  urgency: Urgency;
  immediateSteps: string[];
  rights: string[];
  options: string[];
  documents: string[];
  whereToComplain: HelpEntry[];
  keyDeadlines: string[];
  disclaimer: string;
};

export type LegalSource = {
  id: string;
  chunkId: string;
  domain: string;
  category: string;
  jurisdiction: string;
  year: number;
  title: string;
  citation: string;
  snippet: string;
  passages: string[];
  match: number;
  score: number;
  why: string;
};

export type RetrievalMeta = {
  engine: string;
  index?: string;
  namespace?: string;
  embeddingModel?: string;
  reason?: string;
  candidates?: number;
  latencyMs?: number;
};

export type LlmMeta = {
  used: boolean;
  model: string;
  reason?: string | null;
  latencyMs?: number;
};

export type ClassifierMeta = {
  model: string;
  labels?: string[];
  latencyMs?: number;
};

export type AskMeta = {
  classifier: ClassifierMeta;
  retrieval: RetrievalMeta;
  llm: LlmMeta;
  corpusItems: number;
  latencyMs: number;
};

export type AskResponse = {
  query: string;
  domain: string;
  confidence: number;
  probabilities: Record<string, number>;
  answer: FirstAidAnswer;
  sources: LegalSource[];
  meta: AskMeta;
  generatedAt: string;
};

export type PineconeHealth = {
  configured: boolean;
  index: string;
  namespace: string;
  embeddingModel: string;
  ready: boolean;
  lastError: string | null;
};

export type GroqHealth = {
  configured: boolean;
  model: string;
  ready: boolean;
  lastError: string | null;
};

export type HealthResponse = {
  status: string;
  version: string;
  classifier: {
    loaded: boolean;
    model: string;
    labels: string[];
  };
  rag: {
    pinecone: PineconeHealth;
    groq: GroqHealth;
    retrieval: { engine: string };
    corpusItems: number;
  };
  corpusItems: number;
  timestamp: string;
};

export type ExampleQuery = {
  label: string;
  query: string;
};

export const DEFAULT_DISCLAIMER =
  "Judicia provides legal information and first-aid guidance, not legal advice. " +
  "Laws change and every situation is different. Before making a decision that affects " +
  "your money, liberty, family, or property, speak to a qualified advocate. " +
  "If you or someone else is in immediate danger, call 112.";

export const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    label: "Account hacked",
    query: "Someone hacked my Instagram account and changed my password.",
  },
  {
    label: "Unpaid salary",
    query: "My employer has not paid my salary for three months and terminated me without notice.",
  },
  {
    label: "Security deposit",
    query: "My landlord refuses to return my rental security deposit after I vacated the flat.",
  },
  {
    label: "Wrong product",
    query: "An online store delivered a completely different product and is refusing a refund.",
  },
  {
    label: "Maintenance",
    query: "My husband and I are separated and I need maintenance for our child.",
  },
  {
    label: "Threats",
    query: "A person is repeatedly threatening to harm me and my family.",
  },
];

export const DOMAINS = [
  "Civil",
  "Constitutional",
  "Consumer",
  "Criminal",
  "Cyber",
  "Family",
  "Labour",
  "Property",
];

const API_BASE = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ""
).replace(/\/+$/, "");

type JsonObject = Record<string, unknown>;

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function asObject(value: unknown): JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asStringArray(value: unknown): string[] {
  if (typeof value === "string") {
    return value
      .split(/\s*\|\s*|\n+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return asArray(value)
    .map((item) => asString(item))
    .filter(Boolean);
}

function asUrgency(value: unknown): Urgency {
  const urgency = asString(value).toLowerCase();
  if (urgency === "emergency" || urgency === "urgent" || urgency === "normal") {
    return urgency;
  }
  return "normal";
}

function asProbabilities(value: unknown): Record<string, number> {
  const source = asObject(value);
  const output: Record<string, number> = {};
  for (const [label, raw] of Object.entries(source)) {
    const score = asNumber(raw, Number.NaN);
    if (Number.isFinite(score)) output[label] = score;
  }
  return output;
}

function asHelpEntry(value: unknown): HelpEntry | null {
  if (typeof value === "string") {
    const name = value.trim();
    return name ? { name, detail: "", phone: null, link: null } : null;
  }

  const item = asObject(value);
  const name = asString(item.name || item.title || item.authority);
  if (!name) return null;

  return {
    name,
    detail: asString(item.detail || item.description),
    phone: asString(item.phone) || null,
    link: asString(item.link) || null,
  };
}

function normalizeAnswer(value: unknown): FirstAidAnswer {
  const answer = asObject(value);
  return {
    summary: asString(answer.summary, "Review the steps below and keep your evidence safe."),
    urgency: asUrgency(answer.urgency),
    immediateSteps: asStringArray(answer.immediate_steps).slice(0, 8),
    rights: asStringArray(answer.rights).slice(0, 8),
    options: asStringArray(answer.options).slice(0, 6),
    documents: asStringArray(answer.documents).slice(0, 8),
    whereToComplain: asArray(answer.where_to_complain)
      .map(asHelpEntry)
      .filter((entry): entry is HelpEntry => entry !== null)
      .slice(0, 6),
    keyDeadlines: asStringArray(answer.key_deadlines).slice(0, 4),
    disclaimer: asString(answer.disclaimer, DEFAULT_DISCLAIMER),
  };
}

function normalizeSource(value: unknown): LegalSource {
  const source = asObject(value);
  return {
    id: asString(source.id) || asString(source.chunk_id),
    chunkId: asString(source.chunk_id) || asString(source.id),
    domain: asString(source.domain, "General"),
    category: asString(source.category, "Statute"),
    jurisdiction: asString(source.jurisdiction, "India"),
    year: Math.trunc(asNumber(source.year, 0)),
    title: asString(source.title, "Untitled provision"),
    citation: asString(source.citation),
    snippet: asString(source.snippet),
    passages: asStringArray(source.passages),
    match: Math.min(1, Math.max(0, asNumber(source.match ?? source.relevance, 0))),
    score: asNumber(source.score, 0),
    why: asString(source.why),
  };
}

function normalizeAskResponse(value: unknown): AskResponse {
  const data = asObject(value);
  const meta = asObject(data.meta);
  const sources = (
    asArray(data.sources).length ? asArray(data.sources) : asArray(data.results)
  ).map(normalizeSource);

  return {
    query: asString(data.query, ""),
    domain: asString(data.domain, "General"),
    confidence: Math.min(1, Math.max(0, asNumber(data.confidence, 0))),
    probabilities: asProbabilities(data.probabilities),
    answer: normalizeAnswer(data.answer),
    sources,
    meta: {
      classifier: {
        model: asString(asObject(meta.classifier).model, "judicia-domain-model"),
        labels: asStringArray(asObject(meta.classifier).labels),
        latencyMs: asNumber(asObject(meta.classifier).latency_ms, 0),
      },
      retrieval: {
        engine: asString(asObject(meta.retrieval).engine, "local"),
        index: asString(asObject(meta.retrieval).index) || undefined,
        namespace: asString(asObject(meta.retrieval).namespace) || undefined,
        embeddingModel: asString(asObject(meta.retrieval).embedding_model) || undefined,
        reason: asString(asObject(meta.retrieval).reason) || undefined,
        candidates: asNumber(asObject(meta.retrieval).candidates, sources.length),
        latencyMs: asNumber(asObject(meta.retrieval).latency_ms, 0),
      },
      llm: {
        used: Boolean(asObject(meta.llm).used),
        model: asString(asObject(meta.llm).model, "rule-based playbook"),
        reason: asString(asObject(meta.llm).reason) || null,
        latencyMs: asNumber(asObject(meta.llm).latency_ms, 0),
      },
      corpusItems: asNumber(meta.corpus_items, 0),
      latencyMs: asNumber(meta.latency_ms, 0),
    },
    generatedAt: asString(data.generated_at, new Date().toISOString()),
  };
}

function normalizeHealth(value: unknown): HealthResponse {
  const data = asObject(value);
  const rag = asObject(data.rag);
  const pinecone = asObject(rag.pinecone);
  const groq = asObject(rag.groq);
  const classifier = asObject(data.classifier);

  return {
    status: asString(data.status, "online"),
    version: asString(data.version, "2.0.0"),
    classifier: {
      loaded: Boolean(classifier.loaded),
      model: asString(classifier.model, "judicia-domain-model"),
      labels: asStringArray(classifier.labels),
    },
    rag: {
      pinecone: {
        configured: Boolean(pinecone.configured),
        index: asString(pinecone.index, "judicia-legal"),
        namespace: asString(pinecone.namespace, "legal-corpus"),
        embeddingModel: asString(pinecone.embedding_model, "multilingual-e5-large"),
        ready: Boolean(pinecone.ready),
        lastError: asString(pinecone.last_error) || null,
      },
      groq: {
        configured: Boolean(groq.configured),
        model: asString(groq.model, "llama-3.3-70b-versatile"),
        ready: Boolean(groq.ready),
        lastError: asString(groq.last_error) || null,
      },
      retrieval: {
        engine: asString(asObject(rag.retrieval).engine, "local"),
      },
      corpusItems: asNumber(rag.corpus_items, asNumber(data.corpus_items, 0)),
    },
    corpusItems: asNumber(data.corpus_items, 0),
    timestamp: asString(data.timestamp),
  };
}

export async function askJudicia(
  query: string,
  signal?: AbortSignal
): Promise<AskResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal,
    });

    if (!response.ok) {
      let message = `The server returned an error (${response.status}).`;
      try {
        const body = asObject(await response.json());
        if (asString(body.error)) message = asString(body.error);
      } catch {
        // Keep the status-based message.
      }
      throw new ApiError(message, response.status);
    }

    let raw: unknown;
    try {
      raw = await response.json();
    } catch {
      throw new ApiError("The server returned an unreadable response.", response.status);
    }
    return normalizeAskResponse(raw);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError(
      "Could not reach the Judicia backend. Make sure it is running on port 8000.",
      0
    );
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/health`, { signal });
    if (!response.ok) return null;
    return normalizeHealth(await response.json());
  } catch {
    return null;
  }
}
