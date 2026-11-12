"""Snapshot-local Markdown parsing and lexical retrieval."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from urllib.parse import urlsplit

from contract import Source

DASHES = str.maketrans("–—‑−’\u00a0", "----' ")
LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\([^\n]*?\)")
WORD = re.compile(r"[a-z0-9]+")
MONTHS = r"January|February|March|April|May|June|July|August|September|October|November|December"
WRITTEN_DATE = rf"(?:{MONTHS}) \d{{1,2}},? \d{{4}}"
DATE = rf"(?:\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|{WRITTEN_DATE})"
STOP = set("""a an and any are as at be by can could did do does for from get give have how
i if in is it know me my of on or our please s should so tell than that the their them there
these they this to up us use want was were what when where which who will with would you your
dupaco current today right now about like""".split())
ALIASES = {
    "automobile": "auto", "car": "auto", "cars": "auto", "vehicles": "vehicle",
    "certificates": "certificate", "cd": "certificate", "cds": "certificate",
    "accounts": "account", "rates": "rate", "aprs": "apr", "fees": "fee",
    "branches": "branch", "locations": "location", "loans": "loan",
    "members": "member", "membership": "member", "qualify": "eligible",
    "qualification": "eligible", "qualifications": "eligible", "eligibility": "eligible",
    "months": "month", "years": "year", "recognition": "recognized",
    "change": "update", "updating": "update", "info": "information",
    "options": "option", "wallets": "wallet", "transfers": "transfer",
    "cards": "card", "closing": "close", "closes": "close", "closed": "close",
    "deposits": "deposit", "checks": "check", "bills": "bill", "paid": "pay",
    "paying": "pay", "alerts": "alert", "notification": "alert", "notifications": "alert",
    "install": "download", "mobile": "phone", "repeat": "schedule",
}


def norm(value):
    return re.sub(r"\s+", " ", value.translate(DASHES)).strip().lower()


def clean(value):
    value = IMAGE.sub("", value)
    value = LINK.sub(r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\((?:tel:|mailto:|#)[^)]*\)", r"\1", value)
    value = re.sub(r"\\+[rn]", " ", value)
    value = re.sub(r"\\([*$|])", r"\1", value)
    return re.sub(r"\s+", " ", value.replace("**", "").replace("__", "")).strip()


def tokens(value):
    return [ALIASES.get(word, word) for word in WORD.findall(norm(value)) if word not in STOP]


def words(value):
    return set(tokens(value))


VERIFICATION_CODE = re.compile(
    r"\b(?:two[ -]step(?: verification)?|2[ -]step|sign[ -]in|log[ -]?in|verification|"
    r"one[ -]time(?: sign[ -]in)?) (?:passcodes?|codes?)\b"
)


def search_tokens(value):
    text = VERIFICATION_CODE.sub("verification code", norm(value))
    text = re.sub(r"\bplastic card\b", "physical card", text)
    return tokens(re.sub(r"\b(?:edit|modify)\b", "update", text))


def requested_platform(question):
    text = norm(question)
    web = re.search(r"\b(?:browser|website|web version|shine online)\b", text)
    mobile = re.search(r"\b(?:shine (?:mobile|app)|(?:mobile|phone) app)\b", text)
    if bool(web) == bool(mobile):
        return ""
    return "web" if web else "mobile"


def phrases(sequence):
    return set(zip(sequence, sequence[1:]))


def parse_date(value):
    value = value.replace(",", "")
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%B %d %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def labeled_date(value, prefix):
    match = re.search(prefix + r"\s*(" + DATE + ")", value, re.I)
    return parse_date(match[1]) if match else None


@dataclass(frozen=True)
class Document:
    url: str
    path: str
    title: str
    markdown: str
    snapshot: str
    published: date | None
    updated: date | None = None
    effective: date | None = None

    @classmethod
    def from_page(cls, page, snapshot):
        published = None
        path = urlsplit(page.url).path.rstrip("/") or "/"
        top = "\n".join(page.markdown.splitlines()[:12])
        if path.startswith(("/newsroom/", "/learn/blog/")):
            dates = re.findall(r"^(" + WRITTEN_DATE + r")\s*$", top, re.M)
            if dates:
                published = parse_date(dates[0])
        return cls(
            page.url, path, page.title, page.markdown, snapshot, published,
            updated=labeled_date(top, r"(?m)^Last (?:updated|revised):"),
            effective=labeled_date(top, r"(?m)^Effective (?:date|as of):?"),
        )

    def source(self, quote=None):
        return Source(url=self.url, snapshot_id=self.snapshot, quote=quote)

    def available_on(self, cutoff):
        return all(day is None or day <= cutoff for day in (self.published, self.updated, self.effective))


@dataclass(frozen=True)
class Table:
    document: Document
    context: str
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[tuple[str, ...], str], ...]
    effective: date | None
    expires: date | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Passage:
    document: Document
    heading: str
    text: str
    raw: str
    question: str = ""
    canonical_faq: bool = False
    platform: str = ""


FAQ_LINK = re.compile(r"^\s*[-*]\s+\[([^\]]+\?)\]\(([^)]+)\)\s*$")
TABLE_CELL = re.compile(r"(?<!\\)\|")
LIST_ITEM = re.compile(r"^(?:[-*]|\d+[.)])\s+")
RELATED_HEADING = re.compile(r"handpicked for you|related (?:articles|posts)", re.I)
PLATFORM_LABEL = re.compile(r"\*\*(?:In |On )?Shine (Online|Mobile)(?: Banking)?:?\*\*:?", re.I)


def markdown_lines(markdown):
    # Unicode line separators can be soft breaks inside a captured Markdown paragraph.
    return re.split(r"\r\n?|\n", markdown)


def prose_end(lines, start, in_faq, platform_label=False):
    """Keep a short FAQ answer or a procedure and its contiguous list together."""
    size = len(lines[start])
    end = start + 1
    for index in range(end, min(len(lines), start + 30)):
        line = lines[index].strip()
        if not line:
            continue
        if line.startswith(("#", "|", "![", "---", "* * *", "[", "✕")) or PLATFORM_LABEL.fullmatch(line):
            break
        if not (in_faq or LIST_ITEM.match(line) or (platform_label and end == start + 1)):
            break
        size += len(line)
        if size > 1800:
            break
        end = index + 1
    return end


def markdown_blocks(document):
    """Yield table rows with their local context and short prose passages."""
    lines = markdown_lines(document.markdown)
    headings = {}
    previous = ""
    faq, canonical_faq = "", False
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        heading = re.match(r"^(#{1,6})\s+(.+)", raw)
        if heading:
            level = len(heading[1])
            headings = {depth: name for depth, name in headings.items() if depth < level}
            headings[level] = clean(heading[2])
            previous = ""
            faq = ""
        elif any(RELATED_HEADING.fullmatch(name) for name in headings.values()):
            pass
        elif raw.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|\s*[:-]{2,}", lines[i + 1]):
            start = i
            headers = tuple(clean(cell) for cell in TABLE_CELL.split(raw.strip("|")))
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                quote = lines[i].strip()
                cells = tuple(clean(cell) for cell in TABLE_CELL.split(quote.strip("|")))
                if len(cells) == len(headers):
                    rows.append((cells, quote))
                i += 1
            notes = []
            for line in lines[i:i + 12]:
                if line.lstrip().startswith(("#", "|")):
                    break
                if line.strip():
                    notes.append(line.strip())
                if len(notes) >= 3:
                    break
            note = " ".join(notes)
            effective = labeled_date(note, r"(?<!\d)(?:rates? (?:are )?effective(?: as of)?|effective date:)\s*")
            expires = labeled_date(note, r"\b(?:offer|promotion)\s+(?:ended|ends|expires)(?:\s+on)?")
            yield Table(document, " / ".join(headings.values()), previous, headers, tuple(rows),
                        effective, expires, tuple(notes))
            if rows and not re.search(r"\b(?:APY|APR|rate|yield)\b", " ".join(headers), re.I):
                # Role and policy tables carry meaning in the column headers.
                # Keep them with the evidence when they fit in a single passage.
                table_text = "\n".join(lines[start:i]).strip()
                if len(table_text) <= 2500:
                    yield Passage(document, " / ".join(headings.values()), clean(table_text),
                                  table_text, faq, bool(faq) and canonical_faq)
            previous = ""
            faq = ""
            continue
        elif match := FAQ_LINK.match(raw):
            faq = clean(match[1])
            canonical_faq = urlsplit(match[2]).path.rstrip("/") == document.path
            previous = ""
        elif raw and not raw.startswith(("![", "|", "---", "* * *", "<")):
            label = PLATFORM_LABEL.fullmatch(raw)
            platform = {"online": "web", "mobile": "mobile"}[label[1].lower()] if label else ""
            end = prose_end(lines, i, bool(faq), platform_label=bool(label))
            excerpt = "\n".join(lines[i:end]).strip()
            value = clean(excerpt)
            if value and len(value) <= 2500:
                yield Passage(document, " / ".join(headings.values()), value, excerpt,
                              faq, bool(faq) and canonical_faq, platform)
            faq = ""
            if 2 < len(value) < 105 and not raw.startswith(("-", "* ", "[")) and not value.endswith((".", "?", "!")):
                previous = value
            i = end
            continue
        i += 1


class TextIndex:
    def __init__(self, passages):
        self.passages = tuple(passages)
        self.postings = defaultdict(list)
        self.lengths = []
        self.body_terms = []
        self.context_terms = []
        self.faq_terms = []
        self.passage_phrases = []
        paths = {passage.document.path for passage in self.passages}
        self.path_terms = {path: words(path) for path in paths}
        for i, passage in enumerate(self.passages):
            body = search_tokens(passage.text)
            context = set(search_tokens(passage.heading + " " + passage.document.title))
            faq = search_tokens(passage.question)
            body_set, faq_set = set(body), set(faq)
            self.body_terms.append(body_set)
            self.context_terms.append(context)
            self.faq_terms.append(faq_set)
            self.passage_phrases.append(phrases(body) | phrases(faq))
            for token in body_set | context | faq_set:
                weight = (token in body_set) + 0.55 * (token in context) + 1.2 * (token in faq_set)
                self.postings[token].append((i, weight))
            self.lengths.append(len(body_set))

    def search(self, question, limit=24):
        query_tokens = search_tokens(question)
        query = set(query_tokens)
        query_phrases = phrases(query_tokens)
        platform = requested_platform(question)
        scores = Counter()
        total = len(self.passages)
        for token in sorted(query):
            postings = self.postings.get(token, ())
            idf = min(6, math.log(1 + total / (1 + len(postings))))
            for index, weight in postings:
                scores[index] += idf * weight / (0.8 + 0.012 * self.lengths[index])
        for index in scores:
            passage = self.passages[index]
            scores[index] += 1.4 * len(query_phrases & self.passage_phrases[index])
            if query & self.path_terms[passage.document.path]:
                scores[index] += 1.25
            if passage.canonical_faq and query & self.faq_terms[index]:
                scores[index] += 2.0
            if platform and passage.platform and len(query & self.context_terms[index]) >= 2:
                scores[index] += 7.0 if platform == passage.platform else -7.0
        return [
            (self.passages[index], score, self.body_terms[index], self.context_terms[index], self.faq_terms[index])
            for index, score in scores.most_common(limit)
        ]
