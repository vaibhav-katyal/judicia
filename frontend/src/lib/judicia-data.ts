export type LegalResult = {
  id: string;
  category: string;
  jurisdiction: string;
  year: number;
  title: string;
  citation: string;
  snippet: string;
  passages: string[];
  why: string;
  match: number;
};

export const SUGGESTED_QUERIES = [
  "What is the proportionality test for privacy restrictions?",
  "Damages available for breach of a commercial contract",
  "Are WhatsApp messages admissible as evidence?",
  "Notice requirements before terminating an employee",
];

const CORPUS: LegalResult[] = [
  {
    id: "1",
    category: "Statute",
    jurisdiction: "India",
    year: 2000,
    title: "Section 65B, Indian Evidence Act — Admissibility of Electronic Records",
    citation: "Indian Evidence Act, 1872, s.65B",
    snippet:
      "Any information contained in an electronic record is deemed a document and admissible in evidence without further proof, subject to conditions specified in this section.",
    passages: ["65B(1)", "65B(2)", "65B(4) — Certificate requirement"],
    why: "Directly governs conditions for admitting electronic evidence, including the certificate requirement under 65B(4).",
    match: 0.93,
  },
  {
    id: "2",
    category: "Judgment",
    jurisdiction: "India",
    year: 2020,
    title: "Arjun Panditrao Khotkar v. Kailash Kushanrao Gorantyal",
    citation: "(2020) 7 SCC 1",
    snippet:
      "A certificate under Section 65B(4) is a condition precedent to the admissibility of electronic evidence, clarifying earlier conflicting rulings.",
    passages: ["Para 61", "Para 72"],
    why: "Settles the mandatory nature of the 65B(4) certificate, resolving prior conflicting precedent.",
    match: 0.88,
  },
  {
    id: "3",
    category: "Statute",
    jurisdiction: "India",
    year: 1872,
    title: "Section 73, Indian Contract Act — Compensation for Breach",
    citation: "Indian Contract Act, 1872, s.73",
    snippet:
      "When a contract is broken, the injured party is entitled to compensation for loss or damage caused, naturally arising in the usual course of things.",
    passages: ["s.73", "s.74 — Liquidated damages"],
    why: "Foundational provision on remedies available for breach of contract.",
    match: 0.81,
  },
];

export function searchCorpus(query: string): LegalResult[] {
  const q = query.toLowerCase();
  return CORPUS.filter(
    (r) =>
      r.title.toLowerCase().includes(q) ||
      r.snippet.toLowerCase().includes(q) ||
      r.category.toLowerCase().includes(q) ||
      q.split(" ").some((word) => word.length > 3 && r.snippet.toLowerCase().includes(word))
  ).sort((a, b) => b.match - a.match);
}