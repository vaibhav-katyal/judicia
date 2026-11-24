"""
Judicia legal knowledge base helpers.

This module owns everything that can work without an external service:
  * loading / normalising ``legal_corpus.json``
  * turning corpus entries into Pinecone-ready chunks
  * domain-specific first-aid playbooks
  * a safe, rule-based answer used whenever Groq is unavailable

Keeping a strong local fallback means the product still gives useful first-aid
guidance when API keys are missing, quotas are exhausted, or the network is down.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "legal_corpus.json"

DOMAINS = [
    "Civil",
    "Constitutional",
    "Consumer",
    "Criminal",
    "Cyber",
    "Family",
    "Labour",
    "Property",
]

DEFAULT_DISCLAIMER = (
    "Judicia provides legal information and first-aid guidance, not legal advice. "
    "Laws change and every situation is different. Before making a decision that affects "
    "your money, liberty, family, or property, speak to a qualified advocate. "
    "If you or someone else is in immediate danger, call 112."
)


def _text(value: Any, fallback: str = "") -> str:
    """Return a trimmed string, tolerating None and non-string JSON values."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return fallback


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in re.split(r"\s*\|\s*|\n+", value)]
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in (_text(v) for v in value) if item]


def load_corpus(path: str | Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Load and normalise the statutory corpus. Returns [] if the file is absent."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        return []

    with corpus_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {corpus_path.name}, got {type(raw).__name__}")

    corpus: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        source_id = _text(item.get("id")) or f"source-{index + 1}"
        year = item.get("year", 0)
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = 0

        corpus.append(
            {
                "id": source_id,
                "domain": _text(item.get("domain"), "General"),
                "category": _text(item.get("category"), "Statute"),
                "jurisdiction": _text(item.get("jurisdiction"), "India"),
                "year": year,
                "title": _text(item.get("title")),
                "citation": _text(item.get("citation")),
                "snippet": _text(item.get("snippet")),
                "passages": _string_list(item.get("passages")),
                "keywords": _text(item.get("keywords")),
            }
        )
    return corpus


def chunk_text(item: dict[str, Any]) -> str:
    """Build the text that is embedded and stored in Pinecone for one provision."""
    passages = item.get("passages") or []
    passage_block = "\n".join(f"- {passage}" for passage in passages) or "- Not available"
    return (
        f"Legal provision: {item.get('title', '')}\n"
        f"Citation: {item.get('citation', '')}\n"
        f"Domain: {item.get('domain', '')} | Category: {item.get('category', '')} "
        f"| Jurisdiction: {item.get('jurisdiction', '')} | Year: {item.get('year', '')}\n"
        f"Plain-language summary: {item.get('snippet', '')}\n"
        f"Key provisions:\n{passage_block}\n"
        f"Search terms: {item.get('keywords', '')}"
    )


def build_chunks(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert corpus entries into one rich, metadata-complete chunk per provision."""
    chunks: list[dict[str, Any]] = []
    for item in corpus:
        chunks.append(
            {
                "id": f"{item['id']}::main",
                "source_id": item["id"],
                "domain": item["domain"],
                "category": item["category"],
                "jurisdiction": item["jurisdiction"],
                "year": item["year"],
                "title": item["title"],
                "citation": item["citation"],
                "snippet": item["snippet"],
                "passages": item["passages"],
                "keywords": item["keywords"],
                "text": chunk_text(item),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Domain playbooks
# ---------------------------------------------------------------------------
# These are deliberately practical and India-focused. They are used directly
# when Groq is unavailable and as a safety floor when the LLM output is thin.

PLAYBOOKS: dict[str, dict[str, Any]] = {
    "Cyber": {
        "urgency": "urgent",
        "summary": (
            "This looks like a cyber-law problem. The first priority is to stop any ongoing "
            "loss, preserve digital evidence, and file a cyber complaint quickly."
        ),
        "immediate_steps": [
            "If money was taken or you shared an OTP, call 1930 and your bank's fraud desk immediately. Ask them to freeze the account/card and dispute the transaction.",
            "Change the password from a device you trust, turn on two-factor authentication, and sign out of all other sessions.",
            "Do not delete anything. Take screenshots of the profile, messages, transaction IDs, email headers, and URLs.",
            "Report the account to the platform (Instagram, Facebook, Google, WhatsApp, etc.) and follow its hacked-account recovery process.",
            "File a complaint at cybercrime.gov.in and save the acknowledgement number. For identity theft, stalking, or intimate-image abuse, also visit the nearest cyber police cell.",
            "If a fake profile is using your photos, report it to the platform and keep a record of every report and reply.",
        ],
        "rights": [
            "Section 66C of the IT Act covers identity theft using another person's password, OTP, or digital signature.",
            "Section 66D covers cheating by personation using a computer resource.",
            "Sections 66E and 67 cover privacy violations and publishing obscene or private material.",
            "Sections 43 and 66 cover unauthorised access, hacking, and damage to a computer system.",
            "Electronic records can be proved under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 (earlier Section 65B of the Evidence Act).",
        ],
        "options": [
            "Raise a fraud dispute or chargeback with your bank for unauthorised transactions.",
            "File a cyber-crime complaint and, for serious offences, an FIR at the cyber police cell.",
            "Escalate to the platform's grievance officer for account recovery or a fake profile.",
            "Consider a civil suit for damages or an injunction if reputational harm continues.",
        ],
        "documents": [
            "Screenshots of the hacked account, fake profile, chats, and emails",
            "Bank statement plus transaction IDs / UPI reference numbers",
            "Cyber-crime acknowledgement number and platform complaint IDs",
            "Government ID and proof that the account or phone number is yours",
            "Copy of the FIR, if one has been registered",
        ],
        "where_to_complain": [
            {
                "name": "National Cyber Crime Helpline",
                "detail": "Call immediately for financial cyber fraud so the transaction can be challenged.",
                "link": None,
                "phone": "1930",
            },
            {
                "name": "National Cyber Crime Reporting Portal",
                "detail": "File and track a cyber complaint online.",
                "link": "https://cybercrime.gov.in",
                "phone": None,
            },
            {
                "name": "Your bank's fraud desk",
                "detail": "Ask to freeze the card/account and dispute unauthorised debits.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Local cyber police cell / nearest police station",
                "detail": "For identity theft, threats, stalking, or intimate-image abuse.",
                "link": None,
                "phone": "112",
            },
        ],
        "key_deadlines": [
            "Report financial fraud as soon as possible. The first 24-72 hours give the best chance of freezing the money.",
            "Do not wait for the platform to reply before filing the cyber-crime complaint.",
        ],
    },
    "Criminal": {
        "urgency": "emergency",
        "summary": (
            "This looks like a criminal-law matter. Your safety comes first. If there is a threat "
            "of violence, call 112 now; then preserve evidence and get a written police complaint."
        ),
        "immediate_steps": [
            "If you are in danger right now, call 112 (or 100 for police) and move to a safe place.",
            "Get medical treatment for any injury and ask the hospital for a medico-legal certificate (MLC).",
            "Preserve evidence: photos, videos, CCTV requests, witness names and numbers, threatening messages, call recordings, and any weapon or torn clothing.",
            "Go to the nearest police station and give a written complaint. Ask for a copy of the FIR or complaint receipt. A zero-FIR can be filed at any station.",
            "If the police refuse to register an FIR, send the complaint to the Superintendent of Police or approach the Magistrate.",
            "Do not confront the accused or negotiate alone. Tell a trusted person and keep a safety plan.",
        ],
        "rights": [
            "The Bharatiya Nyaya Sanhita, 2023 covers criminal intimidation, assault, theft, and criminal trespass.",
            "You can file a zero-FIR at any police station, even outside the area where the incident happened.",
            "If police refuse to act, you can complain to the Superintendent of Police or the Magistrate.",
            "Women and children have additional protections, including protection orders and special helplines.",
            "Free legal aid is available through the District Legal Services Authority.",
        ],
        "options": [
            "Register an FIR and cooperate with the investigation.",
            "File a complaint with the Superintendent of Police or Magistrate if the FIR is refused.",
            "Seek a protection order or bail conditions where safety requires it.",
            "Pursue compensation or damages through the criminal court or a civil suit.",
        ],
        "documents": [
            "Government ID and a written timeline of what happened",
            "Medico-legal certificate and medical bills",
            "Photos, videos, CCTV footage, and witness contacts",
            "Threatening messages, call recordings, and screenshots",
            "Copy of the FIR / complaint receipt",
        ],
        "where_to_complain": [
            {
                "name": "Emergency response",
                "detail": "Call for police, ambulance, or immediate danger.",
                "link": None,
                "phone": "112",
            },
            {
                "name": "Nearest police station",
                "detail": "Give a written complaint and ask for a copy of the FIR.",
                "link": None,
                "phone": "100",
            },
            {
                "name": "Women Helpline",
                "detail": "Support for women facing violence, threats, or harassment.",
                "link": None,
                "phone": "181",
            },
            {
                "name": "Childline",
                "detail": "Help for children in danger or distress.",
                "link": None,
                "phone": "1098",
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid and a lawyer for eligible persons.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "Report the incident promptly. Medical evidence and witness memory weaken with delay.",
            "There is usually no limitation period for serious criminal offences, but early action protects the case.",
        ],
    },
    "Property": {
        "urgency": "normal",
        "summary": (
            "This looks like a property or tenancy dispute. The key is to prove your title, "
            "possession, and payments, then put the other side on written notice before filing."
        ),
        "immediate_steps": [
            "Collect the title chain: sale deed, gift/partition deed, mutation records, tax receipts, encumbrance certificate, and proof of possession.",
            "If someone is occupying or encroaching, do not use force. Take dated photos/videos and give a written complaint to the police or tehsildar.",
            "Send a written legal notice to the other side by registered post and email. Keep the receipt and a copy.",
            "For a tenant dispute, read the rent agreement and the applicable state rent law. Approach the rent authority or civil court for eviction or a deposit refund.",
            "For a co-owner or ancestral dispute, try mediation first; otherwise file a partition or injunction suit.",
            "Before buying or selling property, get a title search and check for loans, liens, and pending litigation.",
        ],
        "rights": [
            "Sections 105 and 108 of the Transfer of Property Act, 1882 govern tenancy and the landlord's obligations.",
            "The Specific Relief Act, 1963 provides remedies for possession and injunctions.",
            "The Partition Act and Hindu Succession Act govern shares in jointly owned or ancestral property.",
            "The Indian Registration Act, 1908 governs registration of sale deeds and title documents.",
        ],
        "options": [
            "Send a legal notice and try settlement or mediation.",
            "File a civil suit for possession, partition, injunction, or recovery.",
            "Approach the rent authority/controller for a tenancy dispute.",
            "Give a police complaint if there is criminal trespass or a credible threat.",
        ],
        "documents": [
            "Sale deed, gift deed, partition deed, or rent agreement",
            "Property tax receipts, mutation records, and encumbrance certificate",
            "Bank payment proof and rent/deposit receipts",
            "Dated photos/videos of the property and encroachment",
            "Legal notice, postal receipt, and any replies",
            "Family tree or succession documents for ancestral property",
        ],
        "where_to_complain": [
            {
                "name": "Civil court",
                "detail": "For possession, partition, injunction, and recovery claims.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Rent authority / rent controller",
                "detail": "For eviction, rent, and security-deposit disputes under state rent law.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Tehsildar / SDM office",
                "detail": "For mutation, land records, and encroachment on public/private land.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Police",
                "detail": "If there is criminal trespass, intimidation, or a threat to safety.",
                "link": None,
                "phone": "112",
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid for eligible persons.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "Limitation periods apply to possession and specific-performance claims. Consult a lawyer quickly.",
            "Raise a security-deposit claim in writing immediately after vacating.",
        ],
    },
    "Family": {
        "urgency": "urgent",
        "summary": (
            "This looks like a family-law matter. If there is violence or a threat, get to safety "
            "first. Then preserve financial and family records and ask the court for interim protection "
            "or maintenance early."
        ),
        "immediate_steps": [
            "If there is violence or a threat, call 112 or 181 and get to a safe place. A protection order can be sought under the Domestic Violence Act.",
            "Collect the marriage certificate, IDs, income proof, bank statements, wedding expenses, and evidence of cruelty or abandonment.",
            "For maintenance, file an application in the family court (or before the Magistrate) and ask for interim maintenance.",
            "For custody, document the child's schooling, routine, expenses, and your involvement. Ask for interim custody or visitation.",
            "Try court-annexed mediation or counselling only if it is safe. Do not agree to a settlement without legal advice.",
            "Keep a dated diary of incidents, expenses, and communication. Do not unlawfully block the other parent's access to the child.",
        ],
        "rights": [
            "The Hindu Marriage Act, 1955 provides for divorce, judicial separation, and maintenance.",
            "The Hindu Minority and Guardianship Act governs custody and guardianship of children.",
            "The Protection of Women from Domestic Violence Act, 2005 provides protection, residence, and maintenance orders.",
            "You can seek interim maintenance and interim custody while the main case is pending.",
            "Free legal aid is available through the District Legal Services Authority.",
        ],
        "options": [
            "File for maintenance and/or divorce in the family court.",
            "Seek a protection order and residence order under the Domestic Violence Act.",
            "Ask for interim custody or visitation while the custody case is pending.",
            "Use court-annexed mediation for a safe, structured settlement.",
        ],
        "documents": [
            "Marriage certificate and government IDs",
            "Income proofs, bank statements, and asset details",
            "Children's school and medical records",
            "Photos, messages, and a dated incident diary",
            "FIR / medico-legal certificate if there was violence",
            "List of joint assets and household expenses",
        ],
        "where_to_complain": [
            {
                "name": "Family court",
                "detail": "For divorce, maintenance, custody, and guardianship.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Protection Officer (Domestic Violence Act)",
                "detail": "For protection, residence, and maintenance orders.",
                "link": None,
                "phone": "181",
            },
            {
                "name": "Women Helpline",
                "detail": "Immediate support for women in distress.",
                "link": None,
                "phone": "181",
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid and court assistance.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "Interim maintenance and custody can be sought immediately; do not wait for the final divorce.",
            "Domestic-violence protection can be sought urgently if there is a continuing threat.",
        ],
    },
    "Labour": {
        "urgency": "urgent",
        "summary": (
            "This looks like an employment or labour-law issue. Document everything in writing, "
            "demand your dues from HR, and move quickly to the labour authority if the employer does not respond."
        ),
        "immediate_steps": [
            "Keep the appointment letter, contract, salary slips, bank statements, and all HR emails and messages.",
            "Send a written demand to HR/your employer for unpaid salary, notice pay, dues, and the relieving letter. Keep proof of sending.",
            "Do not sign a forced resignation or a blank settlement. If you must sign, write 'under protest' and keep a copy.",
            "Approach the District Labour Commissioner / labour authority and file a claim for unpaid wages or termination dues.",
            "For maternity, gratuity, or PF issues, file the specific claim with the authority or trust. For wrongful termination, consult a labour lawyer about reinstatement or compensation.",
            "If the company is shutting down or not responding, collect colleague statements and consider a group complaint.",
        ],
        "rights": [
            "The Industrial Disputes Act governs retrenchment, notice, and unfair termination.",
            "The Payment of Wages Act protects timely payment and lawful deductions.",
            "The Maternity Benefit Act, 1961 protects maternity leave and related benefits.",
            "The Payment of Gratuity Act, 1972 provides gratuity after qualifying continuous service.",
            "Employees can approach the labour court or industrial tribunal for redress.",
        ],
        "options": [
            "Send a written demand and try an internal grievance process.",
            "File a claim with the District Labour Commissioner/labour court.",
            "File a PF or gratuity claim with the relevant authority/trust.",
            "Consult a labour lawyer about reinstatement or compensation for wrongful termination.",
        ],
        "documents": [
            "Appointment letter and employment contract",
            "Salary slips and bank statements",
            "Attendance and leave records",
            "Termination letter, resignation pressure, or HR emails",
            "ID proof and details of colleagues/witnesses",
            "Maternity, gratuity, or PF records, if relevant",
        ],
        "where_to_complain": [
            {
                "name": "District Labour Commissioner",
                "detail": "For unpaid wages, termination dues, and labour disputes.",
                "link": "https://shramsuvidha.gov.in",
                "phone": None,
            },
            {
                "name": "Labour court / industrial tribunal",
                "detail": "For adjudication of employment disputes.",
                "link": None,
                "phone": None,
            },
            {
                "name": "EPFO",
                "detail": "For provident-fund and pension grievances.",
                "link": "https://epfindia.gov.in",
                "phone": None,
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid for eligible workers.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "File wage claims quickly; limitation periods apply.",
            "Gratuity is generally payable after five years of continuous service.",
        ],
    },
    "Civil": {
        "urgency": "normal",
        "summary": (
            "This looks like a civil or contractual dispute. Read the agreement carefully, "
            "send a clear written demand, and preserve every document before filing a suit or arbitration."
        ),
        "immediate_steps": [
            "Read the contract/agreement and note the breach, payment terms, notice clause, and dispute-resolution clause.",
            "Send a written legal notice with a clear demand and deadline. Attach invoices and proof, and send it by registered post and email.",
            "Preserve all evidence: contract, invoices, payment receipts, emails, chats, delivery proofs, and witness details.",
            "Try settlement or mediation and put any settlement in writing.",
            "If the contract has an arbitration clause, start arbitration. Otherwise, file a civil suit for recovery or damages.",
            "Consider interim attachment or an injunction if the other side may dispose of assets.",
        ],
        "rights": [
            "Section 73 of the Indian Contract Act, 1872 provides compensation for breach of contract.",
            "Order XXXVII of the Code of Civil Procedure allows a summary suit for debt recovery.",
            "The Specific Relief Act, 1963 provides specific performance and injunctive relief.",
            "The Arbitration and Conciliation Act, 1996 governs arbitration where the contract provides for it.",
        ],
        "options": [
            "Send a legal notice and negotiate a settlement.",
            "Use mediation or arbitration if the contract provides for it.",
            "File a civil suit for recovery, damages, or specific performance.",
            "Seek an interim injunction or attachment where assets are at risk.",
        ],
        "documents": [
            "Signed contract/agreement and amendments",
            "Invoices, bills, and payment receipts",
            "Emails, chats, and delivery/acceptance proof",
            "Legal notice and postal receipt",
            "Ledger/statement of account and witness details",
            "Company/party details for filing",
        ],
        "where_to_complain": [
            {
                "name": "Civil court",
                "detail": "For recovery, damages, specific performance, and injunctions.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Arbitration",
                "detail": "If your contract contains an arbitration clause.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Mediation centre",
                "detail": "For a structured settlement before litigation.",
                "link": None,
                "phone": None,
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid for eligible persons.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "A money claim is generally governed by a three-year limitation period from the breach. Verify this with a lawyer.",
            "Send the demand notice early so the limitation period and negotiation record are clear.",
        ],
    },
    "Consumer": {
        "urgency": "normal",
        "summary": (
            "This looks like a consumer dispute. Start with a written complaint to the seller or platform, "
            "then escalate to the National Consumer Helpline and, if needed, the Consumer Commission."
        ),
        "immediate_steps": [
            "Write to the seller/service provider with your order/invoice details and a clear remedy (refund, replacement, or repair). Keep the reply.",
            "Call 1915 or file a grievance on consumerhelpline.gov.in and save the complaint number.",
            "For e-commerce, escalate to the platform's grievance officer and the National Consumer Helpline before filing.",
            "File a complaint before the District Consumer Commission through e-Daakhil with invoices, warranty, and evidence. You can appear yourself.",
            "For a fraudulent online transaction, also call 1930 and report at cybercrime.gov.in.",
            "Do not accept a one-sided settlement without checking your refund/replacement rights.",
        ],
        "rights": [
            "The Consumer Protection Act, 2019 covers defective goods, deficient services, and unfair trade practices.",
            "The E-Commerce Rules, 2020 require platforms to disclose seller details and provide a grievance officer.",
            "You can seek refund, replacement, compensation, and litigation costs.",
            "A consumer complaint can generally be filed within two years of the cause of action.",
        ],
        "options": [
            "Send a written complaint and negotiate a refund or replacement.",
            "Escalate to the National Consumer Helpline and the platform grievance officer.",
            "File a complaint before the District Consumer Commission via e-Daakhil.",
            "For online fraud, file a cyber-crime complaint in parallel.",
        ],
        "documents": [
            "Invoice/bill and warranty card",
            "Payment proof and order confirmation",
            "Order screenshots, chats, and emails",
            "Photos/video of the defect or wrong product",
            "Repair reports and return/refund records",
            "Complaint numbers from the seller and helpline",
        ],
        "where_to_complain": [
            {
                "name": "National Consumer Helpline",
                "detail": "Call or file a grievance online before approaching the Commission.",
                "link": "https://consumerhelpline.gov.in",
                "phone": "1915",
            },
            {
                "name": "e-Daakhil",
                "detail": "File a consumer complaint online with the Consumer Commission.",
                "link": "https://edaakhil.nic.in",
                "phone": None,
            },
            {
                "name": "District Consumer Commission",
                "detail": "For refund, replacement, compensation, and unfair trade practices.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Cyber crime helpline",
                "detail": "If the transaction was fraudulent.",
                "link": "https://cybercrime.gov.in",
                "phone": "1930",
            },
        ],
        "key_deadlines": [
            "A consumer complaint is generally barred after two years from the cause of action.",
            "Report online payment fraud immediately so the bank can try to freeze the money.",
        ],
    },
    "Constitutional": {
        "urgency": "urgent",
        "summary": (
            "This looks like a constitutional or fundamental-rights issue. Record exactly what the authority did, "
            "ask for written reasons, and move quickly if your liberty or livelihood is affected."
        ),
        "immediate_steps": [
            "Write down exactly what the authority did, when, where, and who was involved. Keep copies of every order or notice.",
            "Ask for the written order or reasons and file an RTI application if information is being withheld.",
            "Send a representation to the authority and keep proof of delivery.",
            "For urgent rights violations, consult a lawyer about a writ petition in the High Court (Article 226) or Supreme Court (Article 32).",
            "For discrimination or harassment by public officials, complain to the relevant commission or the National/State Human Rights Commission.",
            "If there is danger or unlawful detention, call 112 and contact a lawyer or legal-aid clinic immediately.",
        ],
        "rights": [
            "Article 14 guarantees equality before the law and non-discrimination.",
            "Article 19 protects speech, expression, assembly, association, and movement.",
            "Article 21 protects life, personal liberty, and privacy.",
            "Articles 32 and 226 provide writ remedies in the Supreme Court and High Court.",
            "The RTI Act, 2005 gives you a right to information from public authorities.",
        ],
        "options": [
            "File a written representation and an RTI application.",
            "File a writ petition in the High Court or Supreme Court for urgent relief.",
            "Complain to the Human Rights Commission or an appropriate ombudsman.",
            "Use free legal aid through the District Legal Services Authority.",
        ],
        "documents": [
            "Orders, notices, and official correspondence",
            "Government ID and proof of the action against you",
            "Photos, videos, and witness details",
            "RTI application and reply",
            "Representation with postal/email proof",
            "Medical records or FIR, if relevant",
        ],
        "where_to_complain": [
            {
                "name": "High Court",
                "detail": "For writ petitions under Article 226.",
                "link": None,
                "phone": None,
            },
            {
                "name": "Supreme Court",
                "detail": "For writ petitions under Article 32.",
                "link": None,
                "phone": None,
            },
            {
                "name": "National Human Rights Commission",
                "detail": "For complaints against public authorities involving human-rights violations.",
                "link": "https://nhrc.nic.in",
                "phone": None,
            },
            {
                "name": "RTI Online",
                "detail": "File an RTI request with a public authority.",
                "link": "https://rtionline.gov.in",
                "phone": None,
            },
            {
                "name": "District Legal Services Authority",
                "detail": "Free legal aid and lawyer support.",
                "link": "https://nalsa.gov.in",
                "phone": None,
            },
        ],
        "key_deadlines": [
            "Writ remedies should be pursued urgently where liberty or livelihood is affected.",
            "An RTI reply is normally due within 30 days.",
        ],
    },
}

GENERIC_PLAYBOOK: dict[str, Any] = {
    "urgency": "normal",
    "summary": (
        "Your situation needs a careful legal review. Start by writing a clear timeline, "
        "preserving every document, and putting the other side on written notice."
    ),
    "immediate_steps": [
        "Write a short timeline of what happened, with dates, names, amounts, and what you want.",
        "Preserve all documents, messages, receipts, photos, and emails. Do not delete anything.",
        "Identify the person or authority you are dealing with and their correct legal name/address.",
        "Send a written notice or complaint and keep proof of delivery.",
        "Contact the District Legal Services Authority for free legal aid, or consult a qualified advocate.",
        "Act before any deadline mentioned in a notice, contract, or court order.",
    ],
    "rights": [
        "You have the right to be heard fairly and to receive a copy of any order or complaint.",
        "Free legal aid is available through the District Legal Services Authority for eligible persons.",
        "Keep evidence of every payment, message, and official communication.",
    ],
    "options": [
        "Negotiate a settlement and put it in writing.",
        "File a complaint with the relevant authority or court.",
        "Consult a lawyer about urgent interim relief.",
    ],
    "documents": [
        "Government ID",
        "All contracts, notices, receipts, and correspondence",
        "Photos, videos, and witness details",
        "Proof of loss or payment",
        "Any court/authority order",
    ],
    "where_to_complain": [
        {
            "name": "District Legal Services Authority",
            "detail": "Free legal aid and guidance for eligible persons.",
            "link": "https://nalsa.gov.in",
            "phone": None,
        },
        {
            "name": "Local court",
            "detail": "For civil, criminal, or family complaints, depending on the issue.",
            "link": None,
            "phone": None,
        },
        {
            "name": "Emergency response",
            "detail": "If there is immediate danger.",
            "link": None,
            "phone": "112",
        },
    ],
    "key_deadlines": [
        "Limitation periods differ by type of claim. Ask a lawyer about the deadline for your matter.",
    ],
}


_EMERGENCY_RE = re.compile(
    r"\b(?:112|danger(?:ous)?|threat(?:en|ening|s)?|attack(?:ed)?|assault(?:ed)?|"
    r"beat(?:en)?|kill(?:ed|ing)?|murder(?:ed)?|rape|sexual(?:ly)?|molest(?:ed)?|"
    r"kidnap(?:ped)?|abduct(?:ed)?|suicide|self[- ]?harm|violence|violent|weapon|"
    r"knife|gun|detained|locked\s+up|domestic\s+violence|stalking|blackmail|sextortion|"
    r"acid\s+attack|burn(?:ed)?|hostage)\b",
    re.IGNORECASE,
)
_URGENT_RE = re.compile(
    r"\b(?:hacked?|hacking|fraud(?:ulent)?|scam(?:med)?|otp|money|debited|"
    r"stolen|theft|terminated|termination|evict(?:ed|ion)?|deadline|notice|"
    r"cheque\s+bounce|harass(?:ed|ment)?|unpaid|salary|wages|maintenance|"
    r"custody|security\s+deposit|refund)\b",
    re.IGNORECASE,
)


def detect_urgency(query: str, domain: str) -> str:
    """Classify urgency from the user's own words and the predicted domain."""
    if _EMERGENCY_RE.search(query):
        return "emergency"
    if domain in {"Criminal", "Family"} and re.search(
        r"\b(?:violence|threat|hurt|harm|afraid|scared|safe)\b", query, re.IGNORECASE
    ):
        return "emergency"
    if _URGENT_RE.search(query):
        return "urgent"
    return "normal"


def _unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = _text(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _shorten(value: str, limit: int = 260) -> str:
    value = _text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def compose_fallback_answer(
    query: str,
    domain: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a useful, source-grounded answer without calling an LLM."""
    sources = sources or []
    playbook = PLAYBOOKS.get(domain, GENERIC_PLAYBOOK)

    top = sources[0] if sources else None
    if top:
        summary = (
            f"This looks like a {domain.lower()} issue. The closest provision in the knowledge base is "
            f"{top.get('citation') or top.get('title')}: {_shorten(str(top.get('snippet', '')), 220)} "
            "Start with the steps below and keep your evidence safe."
        )
    else:
        summary = str(playbook["summary"])

    rights = list(playbook["rights"])
    for source in sources[:3]:
        citation = _text(source.get("citation")) or _text(source.get("title"))
        snippet = _shorten(_text(source.get("snippet")), 190)
        if citation and snippet:
            rights.append(f"{citation}: {snippet}")

    return {
        "summary": summary,
        "urgency": detect_urgency(query, domain),
        "immediate_steps": list(playbook["immediate_steps"]),
        "rights": _unique(rights, 7),
        "options": list(playbook["options"]),
        "documents": list(playbook["documents"]),
        "where_to_complain": [dict(entry) for entry in playbook["where_to_complain"]],
        "key_deadlines": list(playbook["key_deadlines"]),
        "disclaimer": DEFAULT_DISCLAIMER,
    }


def get_playbook(domain: str) -> dict[str, Any]:
    return PLAYBOOKS.get(domain, GENERIC_PLAYBOOK)
