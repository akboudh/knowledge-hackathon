"""Extraction and resolution of rate and fee observations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import re

from contract import Conflict
from evidence import Passage, Table, clean, labeled_date, norm, words

PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
MONEY = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)(k\b|m\b)?", re.I)
TERM_WORDS = dict(zip(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split(),
    range(1, 21),
))
TERM_WORDS.update(zip("thirty forty fifty sixty seventy eighty ninety".split(), range(30, 100, 10)))
TENS = "|".join(word for word, value in TERM_WORDS.items() if value >= 20 and value % 10 == 0)
UNITS = "|".join(word for word, value in TERM_WORDS.items() if value < 10)
TERM_NUMBER = r"\d+|(?:" + TENS + r")[- ](?:" + UNITS + r")|" + "|".join(TERM_WORDS)
TERM = re.compile(r"\b(" + TERM_NUMBER + r")[ -]*(mo(?:nths?)?|yrs?|years?)\b")
MODEL = re.compile(r"\b((?:19|20)\d\d)\s*-\s*(\d{2,4})\b")
PRODUCTS = {
    "certificate": ("certificate", "term share", "term-share", "fixed jumbo", "cd", "cds"),
    "auto loan": ("auto loan", "auto & motorcycle loan", "vehicle loan", "car loan", "car financing"),
    "home equity refinance": ("home equity refinance",),
    "mortgage": ("fixed conventional", "conventional home purchase", "mortgage", "home purchase", "home loan - purchase", "home loan - refinance"),
    "credit card": ("credit card", "platinum visa", "rewards visa"),
    "checking": ("checking",),
    "savings": ("savings", "money market", "hsa", "investor's choice"),
    "personal loan": ("personal loan", "signature loan"),
    "home equity line of credit": ("heloc", "home equity line"),
    "home equity loan": ("home equity loan", "loan - standard"),
    "home improvement loan": ("home improvement",),
    "motorcycle loan": ("motorcycle",),
    "recreational vehicle loan": ("recreational vehicle", "recreational loan", "rv loan", "boat loan", "boat & rv loan", "atv, snowmobile & jet ski loan"),
    "student loan": ("student loan",),
    "construction loan": ("construction loan",),
}


def classify(text):
    text = re.sub(r"\b(loans|certificates|cards)\b", lambda m: m[1][:-1], norm(text))
    for product, patterns in PRODUCTS.items():
        if any(re.search(r"\b" + re.escape(pattern) + r"\b", text) for pattern in patterns):
            return product
    return ""


def term_months(text):
    match = TERM.search(norm(text))
    if not match:
        return None
    quantity = int(match[1]) if match[1].isdigit() else sum(
        TERM_WORDS[word] for word in match[1].replace("-", " ").split()
    )
    return quantity * (12 if match[2].startswith("y") else 1)


def amounts(text):
    multipliers = {"k": 1000, "m": 1_000_000}
    return [float(m[1].replace(",", "")) * multipliers.get((m[2] or "").lower(), 1)
            for m in MONEY.finditer(text)]


def bounds(text):
    values = amounts(text)
    if len(values) >= 2:
        return values[0], values[1]
    if values and re.search(r"or less|up to|maximum", norm(text)):
        return 0, values[0]
    if values and re.search(r"\+|or more|and (?:up|over)|at least", norm(text)):
        return values[0], float("inf")
    return None


def model_years(text):
    normalized = norm(text)
    match = MODEL.search(normalized)
    if match:
        first, last = map(int, match.groups())
        if last < 100:
            last += first // 100 * 100
        return min(first, last), max(first, last)
    single = re.search(r"\b((?:19|20)\d\d)\b", normalized)
    if single and "model year" in normalized:
        year = int(single[1])
        return (0, year) if "older" in normalized else (year, year)
    return None


def number_range(value):
    return tuple(float(number.rstrip("%").strip()) for number in PERCENT.findall(value))


def metric_for(header):
    header = norm(header)
    if "annual percentage yield" in header or "apy" in header:
        return "APY"
    if "annual percentage rate" in header or "apr" in header:
        return "APR"
    if "dividend rate" in header:
        return "dividend rate"
    if header in {"rate", "interest rate"}:
        return "interest rate"
    return None


def product_words(text):
    text = norm(text)
    text = re.sub(r"\bfirst rate\b", "1st rate", text)
    text = re.sub(r"\bhsa\b", "health savings account", text)
    text = re.sub(r"\binvestor(?:'s|s'?)?\s+choice\b(?!\s+money market)", "investor's choice money market", text)
    return words(text) - {"account"}


def authority(path):
    if (path.startswith(("/calculators", "/switch-and-earn", "/learn/", "/newsroom/", "/promo"))
            or re.search(r"(?:^|[/-])(?:offer|promotion|lp)(?:[/-]|$)", path)):
        return 1
    if path in {"/rates", "/business/rates"}:
        return 3
    return 4


@dataclass(frozen=True)
class Rate:
    product: str
    label: str
    tier: str
    term: int | None
    balance: tuple | None
    years: tuple | None
    metric: str
    value: str
    numeric: tuple
    minimum: str
    qualifier: str
    table: object
    quote: str

    @property
    def business(self):
        return self.table.document.path.startswith("/business") or "business" in words(self.label)

    @property
    def identity(self):
        tier = self.balance if self.balance is not None else norm(self.tier)
        return (*self.comparison_key, tier, self.years)

    @property
    def comparison_key(self):
        label = norm(self.label).removeprefix("business ").removesuffix(" (business)")
        return self.product, label, self.term, self.metric, self.business

    @property
    def rank(self):
        path = self.table.document.path
        primary = {
            "certificate": "/personal/savings-account", "savings": "/personal/savings-account",
            "checking": "/personal/checking-accounts", "auto loan": "/personal/loans/auto-motorcycle-loans",
            "mortgage": "/personal/loans/home-loans", "personal loan": "/personal/loans/personal-loans",
        }
        # Effective dates decide between current product publications of equal authority.
        return authority(path), self.table.effective or date.min, path == primary.get(self.product)

    def source(self):
        return self.table.document.source(self.quote)

    def active(self, cutoff):
        return ((self.table.effective is None or self.table.effective <= cutoff)
                and (self.table.expires is None or self.table.expires >= cutoff))


def extract_rates(table):
    columns = {i: metric for i, h in enumerate(table.headers) if (metric := metric_for(h))}
    if not columns:
        return
    tier_index = next((i for i, header in enumerate(table.headers)
                       if norm(header) in {"balance tier", "balance range", "balance"}), None)
    for cells, quote in table.rows:
        row_label = cells[0]
        context = f"{table.caption} {table.context} {table.document.path.replace('/', ' ').replace('-', ' ')}"
        product = classify(row_label) or classify(table.caption) or classify(context)
        if product == "mortgage" and "home equity refinance" in norm(table.caption):
            product = "home equity refinance"
        if ("ira type" in norm(table.headers[0]) or re.match(r"^(?:roth|traditional)\s*:", norm(row_label))):
            product = "ira certificate" if term_months(row_label) else "ira savings"
        if (not product and re.fullmatch(r"\d+\s+months?", norm(row_label))
                and {"APY", "dividend rate"} <= set(columns.values())
                and any("min. to earn" in norm(header) for header in table.headers)):
            product = "certificate"
        if not product and "model year" in norm(row_label) and "loan program" in norm(table.headers[0]):
            product = "auto loan"
        if not product:
            continue
        term = term_months(" ".join(cells[:min(columns)]))
        tier, label, years = "", row_label, None
        if product in {"auto loan", "motorcycle loan"}:
            years = model_years(row_label)
            if years:
                tier = row_label
                label = product.title()
        elif product in {"mortgage", "home equity refinance"}:
            tier = re.sub(r"^.*?(?=\$)", "", row_label) if "$" in row_label else ""
            label = re.sub(r"\s*\$.*", "", row_label)
        elif product == "credit card":
            if norm(row_label) == "standard":
                label = table.caption if "visa" in norm(table.caption) else "Rewards Visa"
            elif "introductory" in norm(row_label):
                label = "Platinum Visa introductory balance transfer"
        elif product == "personal loan":
            label = "Personal Loan"
        elif tier_index is not None:
            tier = cells[tier_index]
            if tier_index == 0:
                label = table.caption or product.title()
        elif "$" in row_label and product not in {"certificate", "ira certificate"}:
            tier = row_label
            label = table.caption or product.title()
        if product == "certificate" and re.fullmatch(r"\d+\s+months?", norm(label)):
            label += " Certificate"
        if table.document.path.startswith("/business") and "business" not in words(label):
            label = "Business " + label
        minimum = next((cells[i] for i, h in enumerate(table.headers) if "min. to earn" in norm(h)), "")
        for index, metric in columns.items():
            figures = PERCENT.findall(cells[index])
            if not figures or len(figures) > 2:
                continue
            yield Rate(
                product, label, tier, term, bounds(tier), years, metric,
                "–".join(v.replace(" ", "") for v in figures),
                number_range(cells[index]), minimum,
                "as low as" if "as low as" in norm(table.headers[index]) else "",
                table, quote,
            )


def extract_prose_rate(passage: Passage):
    """Handle explicitly labeled rate lines when a publisher does not use tables."""
    text = norm(passage.text)
    match = re.match(
        r"^(.{5,95}?)\s+(apy|apr|dividend rate|interest rate)\s*:\s*(\d+(?:\.\d+)?\s*%)",
        text,
    )
    if not match:
        return
    label = passage.text[:match.end(1)]
    product = classify(label)
    if not product:
        return
    term = term_months(label)
    if product == "certificate" and term and not re.search(r"special|jumbo", label, re.I):
        label = f"{term} Month Certificate"
    expiry = labeled_date(text, r"\b(?:offer|promotion)\s+(?:ended|ends|expires)(?:\s+on)?")
    effective = labeled_date(text, r"\beffective(?: as of)?")
    metadata = Table(passage.document, passage.heading, "", (), (), effective, expiry)
    metric = match[2].upper() if match[2] in {"apy", "apr"} else match[2]
    yield Rate(product, label, "", term, None, None, metric, match[3],
               number_range(match[3]), "", "", metadata, passage.raw)


def same_observation(first, second):
    if first.identity == second.identity:
        return True
    if first.years and second.years and first.comparison_key == second.comparison_key:
        return max(first.years[0], second.years[0]) <= min(first.years[1], second.years[1])
    return False


def resolve(group):
    """Return the best applicable observation, abstaining on an equally ranked disagreement."""
    ordered = sorted(group, key=lambda item: (item.rank, item.table.document.url, item.quote), reverse=True)
    best = ordered[0]
    return None if any(item.rank == best.rank and item.numeric != best.numeric for item in ordered[1:]) else best


def distinct_rates(rates):
    groups = defaultdict(list)
    for rate in rates:
        groups[rate.identity].append(rate)
    return [best for group in groups.values() if (best := resolve(group)) is not None]


def conflict_reason(winner):
    if winner is None:
        return "Equally ranked sources disagree."
    if winner.table.effective:
        return f"Preferred the applicable product rate table, effective {winner.table.effective.isoformat()}."
    return "Preferred the applicable product rate table over lower-authority evidence."


def comparable_timelines(versions):
    """Group by product, metric and tier, splitting overlapping model-year ranges."""
    fixed, vehicles = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for version in versions:
        for rate in version.rates:
            if not rate.active(version.day):
                continue
            if rate.years:
                vehicles[rate.comparison_key].append((version.day, rate))
            else:
                fixed[rate.identity][version.day].append(rate)
    for snapshots in fixed.values():
        first = next(iter(snapshots.values()))[0]
        yield first.tier, snapshots
    for observations in vehicles.values():
        boundaries = sorted({bound for _, rate in observations for bound in (rate.years[0], rate.years[1] + 1)})
        for low, end in zip(boundaries, boundaries[1:]):
            snapshots = defaultdict(list)
            for day, rate in observations:
                if rate.years[0] <= low and rate.years[1] >= end - 1:
                    snapshots[day].append(rate)
            if snapshots:
                qualifier = f"model years {low}–{end - 1}" if low else f"model years up to {end - 1}"
                yield qualifier, snapshots


def rate_conflicts(versions):
    """Report same-crawl disagreements separately from supported historical changes."""
    versions = list(versions)
    if not versions:
        return []
    current_day = versions[-1].day
    conflicts = []
    for qualifier, snapshots in comparable_timelines(versions):
        winners, disagreements = {}, {}
        for day, observations in sorted(snapshots.items()):
            official = [rate for rate in observations if rate.rank[0] >= 3]
            if official:
                winner = resolve(official)
                if winner:
                    winners[day] = winner
            values = {}
            for rate in sorted(observations, key=lambda rate: rate.rank, reverse=True):
                values.setdefault(rate.numeric, rate)
            if len(values) > 1:
                disagreements[day] = values

        current = winners.get(current_day)
        if disagreements:
            day = max(disagreements)
            values = disagreements[day]
            observed = list(values.values())
            if current and all(rate.quote != current.quote or rate.table.document.snapshot != current.table.document.snapshot
                               for rate in observed):
                observed.append(current)
            reason = f"Sources published different values in the {day.isoformat()} crawl. "
            if current:
                reason += (conflict_reason(current) if day == current_day else
                           f"The latest crawl ({current_day.isoformat()}) supports the reported resolution.")
            else:
                reason += "The latest crawl does not support a single authoritative resolution for this tier."
            conflicts.append(make_conflict(observed, qualifier, current, reason))

        history = {rate.numeric: rate for rate in winners.values()}
        if len(history) > 1:
            first_day, last_day = min(winners), max(winners)
            reason = (f"Authoritative observations changed between the {first_day.isoformat()} "
                      f"and {last_day.isoformat()} crawls. ")
            reason += ("The resolution is the latest supported value." if current else
                       "The latest crawl does not support a current value for this tier.")
            conflicts.append(make_conflict(list(history.values()), qualifier, current, reason))
    return conflicts


def make_conflict(rates, qualifier, winner, reason):
    first = rates[0]
    label = first.label
    if first.term and not term_months(label):
        label = f"{first.term}-month {label}"
    topic = " ".join(part for part in (label, qualifier, first.metric) if part)
    values = list({rate.numeric: rate.value for rate in rates}.values())
    return Conflict(topic, values, [rate.source() for rate in rates], winner.value if winner else None, reason)


@dataclass(frozen=True)
class Fee:
    label: str
    detail: str
    document: object
    effective: date | None
    raw: str


def extract_fees(document):
    if "fee-schedule" not in document.path and not re.search(r"/fees(?:/|$)", document.path):
        return
    effective = labeled_date(document.markdown[:600], r"\bEffective")
    lines = document.markdown.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) > 250 or line.startswith(("#", "*", "!", "[")):
            continue
        continuation = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if "$" not in line and continuation.startswith("(") and "$" in continuation:
            line += "\n" + continuation
        value = clean(line.strip("|").replace("|", " "))
        match = MONEY.search(value) or re.search(r"\b(?:free|no charge|price varies)\b", value, re.I)
        if match and len(words(value[:match.start()])) >= 2:
            label = value[:match.start()].strip(" :–-")
            yield Fee(label, f"{label} {value[match.start():]}", document, effective, line)
