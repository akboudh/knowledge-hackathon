"""Entry point loaded by the participant kit's scoring harness."""
from bisect import bisect_right
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime
import re

from contract import Answer, Snapshot
from evidence import MONTHS, Document, Passage, Table, TextIndex, norm, parse_date, words
from evidence import markdown_blocks
from rates import amounts, authority, bounds, classify, conflict_reason, distinct_rates, product_words
from rates import extract_fees, extract_prose_rate, extract_rates, rate_conflicts, resolve, same_observation, term_months
from service import answer_service, service_conflicts
from rag import Rag


@dataclass(frozen=True)
class Version:
    day: date
    documents: dict
    index: TextIndex
    rates: tuple
    fees: tuple


def unknown(topic=None):
    text = f"I can't confirm {topic} from the supplied website records." if topic else (
        "I don't know from the supplied website records. Please contact the credit union to confirm."
    )
    return Answer(text=text, confident=False)


MONTH_NAME = rf"(?:{MONTHS}|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)"
QUERY_DATE = re.compile(
    rf"(?P<exact>\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})"
    rf"|\b(?P<month>{MONTH_NAME})\.?(?:\s+(?P<day>\d{{1,2}})(?!\d)(?:st|nd|rd|th)?)?"
    rf"(?:,?\s+(?P<year>\d{{4}}))?\b",
    re.I,
)


def query_date(match, year):
    if match["exact"]:
        exact = parse_date(match["exact"])
        if exact is None:
            raise ValueError("Invalid comparison date")
        return exact
    month = datetime.strptime(match["month"][:3], "%b").month
    year = int(match["year"] or year)
    day = int(match["day"]) if match["day"] else monthrange(year, month)[1]
    return date(year, month, day)


def inline_date(question, latest):
    for match in QUERY_DATE.finditer(question):
        if not re.search(r"\b(?:as of|on|in|during|from|the)\s*$", question[:match.start()], re.I):
            continue
        # A yearless named month can also be an ordinary verb.
        if match["exact"] or match["day"] or match["year"]:
            return query_date(match, latest.year)
    return None


@dataclass(frozen=True)
class RateQuery:
    product: str
    terms: set
    term: int | None
    balance: tuple | None
    model_year: int | None
    metric: str | None
    business: bool | None
    stated_rate: float | None

    @classmethod
    def parse(cls, question):
        terms = words(question)
        metric = next((metric for expression, metric in (
            (r"\b(?:apys?|annual percentage yields?)\b", "APY"),
            (r"\b(?:aprs?|annual percentage rates?)\b", "APR"),
            (r"\bdividend (?:rate|interest)\b", "dividend rate"),
        ) if re.search(expression, question)), None)
        asks_for_value = re.search(
            r"\b(?:rates?|yields?)\b|\b(?:earn|pay|offer)s?\s+\d+(?:\.\d+)?\s*%"
            r"|\b(?:how much|what percentage)\b.{0,60}\b(?:earn|pay)\b",
            question,
        )
        if not metric and not asks_for_value:
            return None
        product = classify(question)
        term = term_months(question)
        if not product and re.search(r"\b(?:visa|card) aprs?\b", question):
            product = "credit card"
        if not product and term and terms & {"auto", "vehicle", "truck"}:
            product = "auto loan"
        if not product:
            return None
        if product == "certificate" and "ira" in terms:
            product = "ira certificate"
        if not metric:
            if product in {"certificate", "ira certificate", "checking"}:
                metric = "APY"
            elif product in {"mortgage", "home equity refinance"} and "interest rate" in question:
                metric = "interest rate"
        balance = bounds(question)
        requested_amounts = amounts(question)
        if not balance and requested_amounts:
            balance = (requested_amounts[0], requested_amounts[0])
        model_question = QUERY_DATE.sub("", question)
        model = re.search(r"\b((?:19|20)\d{2})\b", model_question)
        year = int(model[1]) if model and product in {"auto loan", "motorcycle loan"} else None
        business = "business" in terms
        if product == "credit card" and not terms & {"business", "personal", "platinum", "rewards"}:
            business = None
        percentage = re.search(r"(\d+(?:\.\d+)?)\s*%", question)
        stated_rate = float(percentage[1]) if percentage else None
        return cls(product, terms, term, balance, year, metric, business, stated_rate)


def rate_request(question, version, cutoff):
    query = RateQuery.parse(question)
    if not query:
        return None
    product, balance = query.product, query.balance
    if product == "mortgage":
        mortgage_type = re.search(r"adjustable|\barm\b|\b(?:fha|usda|va|jumbo)\b|\d+\s*/\s*\d+", question)
        if mortgage_type:
            return unknown("a published mortgage rate for that program")
    active = [rate for rate in version.rates if rate.product == product and rate.active(cutoff)]
    filtered = [
        rate for rate in active
        if (query.term is None or query.term == rate.term)
        and (query.metric is None or query.metric == rate.metric)
        and (query.business is None or query.business == rate.business)
        and (query.model_year is None or (rate.years and rate.years[0] <= query.model_year <= rate.years[1]))
    ]
    if product in {"savings", "checking"}:
        labels = {rate.label for rate in active}
        names = {label: product_words(label) - {"business"} for label in labels}
        requested_name = product_words(question) - {"business"}
        named = {
            label: len(tokens) for label, tokens in names.items()
            if tokens - {product} and tokens <= requested_name
        }
        if named:
            specificity = max(named.values())
            filtered = [rate for rate in filtered if named.get(rate.label) == specificity]
        elif product == "savings" and {"money", "market"} <= requested_name:
            filtered = [rate for rate in filtered if {"money", "market"} <= names[rate.label]]
        elif product == "savings" and query.stated_rate is None:
            filtered = [rate for rate in filtered if norm(rate.label).endswith("share savings")]
    if balance and product in {"savings", "checking", "mortgage", "home improvement loan", "recreational vehicle loan"}:
        filtered = [rate for rate in filtered if rate.balance and
                    rate.balance[0] <= balance[0] <= balance[1] <= rate.balance[1]]
    if product in {"certificate", "ira certificate"}:
        explicit_standard = bool(re.search(r"\b(?:non[ -]?jumbo|standard|regular)\b", question))
        if explicit_standard or "jumbo" in query.terms:
            jumbo = not explicit_standard
            filtered = [rate for rate in filtered if ("jumbo" in words(rate.label)) == jumbo]
        if balance and "minimum" in query.terms:
            filtered = [rate for rate in filtered if amounts(rate.minimum) == [balance[0]]]
        if "special" in query.terms:
            filtered = [rate for rate in filtered if "special" in norm(rate.label)]
        elif "regular" in query.terms or (query.term and any("special" not in norm(rate.label) for rate in filtered)):
            filtered = [rate for rate in filtered if "special" not in norm(rate.label)]
    if product == "credit card":
        intro = bool(re.search(r"\b(?:intro(?:ductory)?|balance transfer)\b", question)) and "standard" not in question
        filtered = [rate for rate in filtered if ("introductory" in norm(rate.label)) == intro]
        brand = next((brand for brand in ("platinum", "rewards") if brand in query.terms), None)
        if brand:
            filtered = [rate for rate in filtered if brand in norm(rate.label)]
    if product in {"mortgage", "home equity refinance"}:
        purpose = "refinance" if product == "home equity refinance" or re.search(r"\brefi(?:nance|nancing)?\b", question) else "purchase"
        filtered = [rate for rate in filtered if norm(rate.label).startswith(purpose)]
        if balance and filtered:
            smallest = min(rate.balance[1] for rate in filtered)
            filtered = [rate for rate in filtered if rate.balance[1] == smallest]
    # When official tables cover a product, legacy calculators and marketing
    # tables remain comparison evidence. They cannot fill missing official tiers.
    official = [rate for rate in filtered if rate.rank[0] >= 3]
    if not official:
        return unknown("a published rate for those terms")
    if product in {"auto loan", "motorcycle loan"} and query.model_year is not None and query.term is not None:
        resolved = resolve(official)
        chosen = [resolved] if resolved else []
    else:
        chosen = distinct_rates(official)
    if not chosen:
        return unknown("a single supported rate")
    if product == "savings" and re.search(r"\b(?:highest|maximum|top|best)\b", question):
        best_yield = max(rate.numeric[0] for rate in chosen)
        chosen = [rate for rate in chosen if rate.numeric[0] == best_yield]
    if query.stated_rate is not None and re.match(r"^(?:do|does|is|are|can|did)\b", question):
        applicable = [rate for rate in chosen if query.stated_rate in rate.numeric]
        if applicable:
            share = [rate for rate in chosen if norm(rate.label).endswith("share savings")]
            if product == "savings" and "share" not in query.terms and share:
                maximum = max(share, key=lambda rate: rate.numeric)
                if maximum not in applicable:
                    return format_rates(
                        [*applicable, maximum], filtered,
                        "Yes. The matching tiers and the Share Savings maximum are shown below. ",
                    )
            return format_rates(applicable, filtered, "Yes. ")
        best = max(rate.numeric[-1] for rate in chosen)
        return format_rates([rate for rate in chosen if rate.numeric[-1] == best], filtered,
                            "No. The highest applicable published rate is shown below. ")
    return format_rates(chosen, filtered)


def format_rates(chosen, candidates, prefix=""):
    groups = {}
    notes = []
    for rate in sorted(chosen, key=lambda r: (r.term or 0, norm(r.label), r.balance or (0, 0), r.tier, r.metric)):
        label = rate.label
        if rate.term and not term_months(label):
            label += f", {rate.term} months"
        if rate.tier:
            label += f", {rate.tier}"
        if rate.minimum and rate.product in {"certificate", "ira certificate"}:
            label += f" ({rate.minimum} minimum to earn)"
        item = f"{rate.qualifier} {rate.value} {rate.metric}".strip()
        groups.setdefault(label, []).append((item, rate.source()))
        if any(rate.numeric != other.numeric and same_observation(rate, other) for other in candidates):
            notes.append(conflict_reason(rate))
    lines = [f"{label}: {', '.join(item for item, _ in entries)}." for label, entries in groups.items()]
    sources = {}
    for entries in groups.values():
        for _, source in entries:
            sources.setdefault((source.url, source.quote), source)
    return Answer(prefix + "\n".join(lines), list(sources.values()),
                  conflict_note=" ".join(dict.fromkeys(notes)) or None)


def comparison_dates(question, latest):
    if not re.search(r"\b(?:changed?|compare|difference|versus|vs\.?|increased?|decreased?|went)\b", question, re.I):
        return []
    if not re.search(r"\b(?:between|since|from|versus|vs\.?|compare)\b", question, re.I):
        return []
    matches = [
        match for match in QUERY_DATE.finditer(question)
        if match["exact"] or match["day"] or match["year"]
        or match["month"].lower() in MONTHS.lower().split("|")
    ]
    trailing_year = next((int(match["year"]) for match in reversed(matches) if match["year"]), latest.year)
    dates = [query_date(match, trailing_year) for match in matches]
    if len(dates) == 1 and re.search(r"\b(?:since|now|today|current)\b", question, re.I):
        dates.append(latest)
    return dates[:2] if len(dates) >= 2 else []


def fee_request(question, version, cutoff):
    if not words(question) & {"fee", "cost", "charge"} and "how much" not in question:
        return None
    terms = words(question) - {"fee", "cost", "charge", "much", "pay", "account", "applies"}
    if "wire" in terms:
        if terms & {"receive", "receiving", "received", "incoming"}:
            terms -= {"receive", "receiving", "received"}
            terms.add("incoming")
        if terms & {"overseas", "abroad", "international"}:
            terms -= {"overseas", "abroad"}
            terms.add("international")
    selectors = []
    if "wire" in terms:
        selectors = [{"wire", direction} for direction in ("domestic", "incoming", "international")
                     if direction in terms]
    elif {"atm", "non"} <= terms:
        selectors = [{"non", "atm", "usage"}]
    elif {"rush", "debit", "card"} <= terms:
        selectors = [{"rush", "debit", "card"}]
    elif "bill" in terms and terms & {"online", "statements"}:
        selectors = [{"per", "bill", "pay"}]
    active = [fee for fee in version.fees if not fee.effective or fee.effective <= cutoff]
    if selectors:
        rows = []
        for required in selectors:
            candidates = [fee for fee in active if required <= words(fee.label)]
            if not candidates:
                return None
            newest = max(fee.effective or date.min for fee in candidates)
            matching = [fee for fee in candidates if (fee.effective or date.min) == newest]
            # Mirror schedules may format the same row differently.
            if len({tuple(amounts(fee.detail)) for fee in matching}) != 1:
                return unknown("a single applicable fee")
            rows.append(min(matching, key=lambda fee: (fee.raw.startswith("|"), fee.document.url)))
        return Answer("\n".join(fee.detail for fee in rows),
                      [fee.document.source(fee.raw) for fee in rows])
    ranked = []
    for fee in active:
        label = words(fee.label)
        overlap = terms & label
        if not overlap or len(overlap) < min(2, len(terms), len(label)):
            continue
        # Require specific words such as "international" or "replacement" to match.
        if terms - label - {"send", "sending", "bank", "price", "transaction"}:
            continue
        rank = (len(overlap) / max(1, len(label)), fee.effective or date.min)
        ranked.append((rank, fee))
    if not ranked:
        return None
    best_rank = max(rank for rank, _ in ranked)
    best = [fee for rank, fee in ranked if rank == best_rank]
    if len({norm(fee.detail) for fee in best}) > 1:
        return unknown("a single applicable fee")
    fee = best[0]
    return Answer(fee.detail, [fee.document.source(fee.raw)])


def prose_answer(question, version, cutoff, rag):
    query = words(question)
    if not query:
        return unknown()
    if rag is not None:
        answer = rag.answer(question, version, cutoff)
        if answer is not None:
            return answer
    for passage, score, body, context, faq in version.index.search(question, limit=35):
        page = passage.document
        if not page.available_on(cutoff):
            continue
        matched = query & (body | context | faq)
        faq_match = len(query & faq) >= 2 and len(query & faq) >= 0.65 * len(faq)
        if (len(matched) < 2 or (len(matched) / len(query) < 0.55 and not faq_match)
                or not (query & body) or score < 6 or len(passage.text) < 40 or passage.text.endswith("?")
                or passage.raw.startswith("[") or authority(page.path) == 1):
            continue
        if len(passage.text) > 530:
            continue
        return Answer(passage.text, [page.source(passage.raw)])
    return unknown()


class Knowledge:
    def __init__(self, rag=None):
        self.versions = {}
        self.rag = rag

    def ingest(self, snapshot: Snapshot):
        day = date.fromisoformat(snapshot.taken_on)
        documents, passages, rates, fees = {}, [], [], []
        for page in snapshot.pages:
            document = Document.from_page(page, snapshot.snapshot_id)
            documents[document.path] = document
            for block in markdown_blocks(document):
                if isinstance(block, Table):
                    rates.extend(extract_rates(block))
                elif isinstance(block, Passage):
                    passages.append(block)
                    if re.search(r"\b(?:apy|apr|dividend rate|interest rate)\s*:", block.text, re.I):
                        rates.extend(extract_prose_rate(block))
            fees.extend(extract_fees(document))
        self.versions[day] = Version(day, documents, TextIndex(passages), tuple(rates), tuple(fees))
        if self.rag is not None:
            self.rag.ingest(day, passages)

    def answer(self, question, as_of=None):
        if not self.versions:
            return unknown()
        days = sorted(self.versions)
        try:
            compare = comparison_dates(question, days[-1]) if not as_of else []
            if compare:
                return self.compare_rates(question, days, compare)
            requested = date.fromisoformat(as_of) if as_of else inline_date(question, days[-1])
        except ValueError:
            return unknown("the requested date")
        cutoff = min(requested or days[-1], days[-1])
        position = bisect_right(days, cutoff)
        if not position:
            return unknown("information before the first website snapshot")
        version = self.versions[days[position - 1]]
        original_question = question
        question = norm(question)
        service = answer_service(question, version, cutoff)
        if service is not None:
            return service
        fee = fee_request(question, version, cutoff)
        if fee is not None:
            return fee
        rate = rate_request(question, version, cutoff)
        return rate if rate is not None else prose_answer(original_question, version, cutoff, self.rag)

    def compare_rates(self, question, days, requested_dates):
        results = []
        for requested in requested_dates:
            cutoff = min(requested, days[-1])
            position = bisect_right(days, cutoff)
            if not position:
                return unknown("both dates in that comparison")
            version = self.versions[days[position - 1]]
            rate = rate_request(norm(question), version, cutoff)
            if not rate or not rate.confident:
                return unknown("a comparable published rate at both dates")
            results.append((version.day, rate))
        (earlier_day, earlier), (later_day, later) = sorted(results, key=lambda item: item[0])
        if earlier_day == later_day:
            return Answer(f"Both dates use the {earlier_day.isoformat()} snapshot.\n{earlier.text}", earlier.sources)
        return Answer(f"Snapshot {earlier_day.isoformat()}\n{earlier.text}\n"
                      f"Snapshot {later_day.isoformat()}\n{later.text}",
                      earlier.sources + later.sources)

    def conflicts(self):
        versions = [self.versions[day] for day in sorted(self.versions)]
        return rate_conflicts(versions) + (service_conflicts(versions[-1]) if versions else [])


def build():
    return Knowledge(Rag.configured())
