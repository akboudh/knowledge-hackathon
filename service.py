"""Extract structured service information from the corresponding website pages."""
from __future__ import annotations

from datetime import date
import re

from contract import Answer, Conflict
from evidence import LINK, MONTHS, clean, labeled_date, norm, words


def cite(document, *quotes):
    return [document.source(quote.strip()) for quote in dict.fromkeys(quotes) if quote.strip()]


def answer_line(document, pattern):
    if document:
        for line in document.markdown.splitlines():
            text = clean(line)
            if re.search(pattern, text, re.I) and not line.lstrip().startswith(("#", "!")):
                return Answer(text.lstrip("- "), cite(document, line))
    return None


def articles(version, cutoff, *path_terms):
    pages = (doc for doc in version.documents.values()
             if doc.published and doc.published <= cutoff and all(term in doc.path for term in path_terms))
    return sorted(pages, key=lambda doc: (doc.published, doc.path), reverse=True)


def find_link(document, predicate):
    if document:
        for label, url in LINK.findall(document.markdown):
            if predicate(label.lower(), url):
                return Answer(url, cite(document, f"[{label}]({url})"))
    return None


SCHEDULE_LABEL = re.compile(r"^\s*\*{2}(Lobby|Drive[- ]?up|Video Teller)\*{2}(.*)", re.I)
DAY_LINE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]+(?:\s*[-–]\s*[a-z]+)?\s*:", re.I)


def schedules(document, kind):
    lines = document.markdown.splitlines()
    for index, raw in enumerate(lines):
        match = SCHEDULE_LABEL.match(raw)
        if not match or kind != norm(match[1]).replace("-", " "):
            continue
        inline = match[2].strip(" :")
        entries = [inline] if DAY_LINE.match(inline) else []
        for line in lines[index + 1:index + 12]:
            text = clean(line).strip(" -")
            if text and not DAY_LINE.match(text):
                break
            if text:
                entries.append(text)
        if entries:
            return entries


BRANCH_ADDRESS = re.compile(r"^(\d[^\n]+?, [A-Za-z .'-]+, [A-Z]{2},? \d{5})\s*$", re.M)


def branch_selector(question):
    # Service words are common across questions. Remaining words can name a branch or city.
    noise = set("""lobby drive through hours hour operating open close schedule day weekday weekend saturday
    sunday friday monday tuesday wednesday thursday time address located location branch bank banking
    general different same all exact nearest near visit find keep common normal typically typical around
    each every still atm teller window street avenue road boulevard today tomorrow today early late
    opening business walk counter listed""".split())
    question = re.sub(rf"\b(?:{MONTHS})\s+(?:\d{{1,2}},?\s+)?\d{{4}}\b", "", question, flags=re.I)
    return words(question) - noise


def branch_hours(question, version, cutoff, contact):
    kind = "drive up" if re.search(r"drive[ -]?(?:up|through|thru)", question) else "lobby"
    selector = branch_selector(question)
    examples = []
    for page in articles(version, cutoff):
        if not re.search(r"\b(?:branch|learning lab)\b", page.title, re.I):
            continue
        if selector and not selector <= words(page.title + " " + page.path):
            continue
        entries = schedules(page, kind)
        if entries:
            examples.append((page, entries))
    if not examples:
        return Answer("I can't confirm hours for that branch in the supplied website records.", confident=False)
    page, entries = examples[0]
    is_general = not selector
    if is_general:
        # Prefer a complete schedule with a weekend example when the branch is unspecified.
        page, entries = next(((doc, lines) for doc, lines in examples if any(
            re.search(r"Saturday:.*(?:noon|12:30|1 p\.m\.)", line, re.I) for line in lines
        )), examples[0])
    subject = (f"{kind.capitalize()} hours vary by branch. One published example is"
               if is_general else f"Published {kind} hours for this branch were")
    text = f"{subject} " + ". ".join(entry.rstrip(".") for entry in entries) + "."
    sources = cite(page, *entries)
    if is_general:
        alternate = next(((doc, lines) for doc, lines in examples if any(
            "noon" in line.lower() for line in lines) and lines != entries), None)
        if alternate and "noon" not in text.lower():
            other, other_entries = alternate
            saturday = next(line for line in other_entries if "noon" in line.lower())
            text += " Another published schedule lists " + saturday.replace(":", " from", 1).rstrip(".") + "."
            sources += cite(other, saturday)
        text += " Which branch would you like to check?"
    if contact and is_general:
        sources += cite(contact, "Stop by and see us! Find your closest branch, ATM or live video teller. We can’t wait to help!")
    return Answer(text, sources)


def branch_address(question, contact):
    if not contact:
        return None
    selector = branch_selector(question)
    if not selector:
        return None
    found = [match[1] for match in BRANCH_ADDRESS.finditer(contact.markdown)
             if selector <= words(match[1])]
    if len(found) == 1:
        return Answer(found[0] + ".", cite(contact, found[0]))
    if len(found) > 1:
        return Answer("I found several matching locations. Which street or neighborhood do you mean?", confident=False)
    return Answer("I can't confirm that branch address in the supplied website records.", confident=False)


def branch_closure(question, version, cutoff):
    selector = branch_selector(question) - {"full", "service", "go", "member", "alternative", "instead"}
    for page in articles(version, cutoff, "clos"):
        if not selector or not selector <= words(page.title + " " + page.path):
            continue
        if re.search(r"\b(?:where|alternative|instead)\b", question):
            return answer_line(page, r"full.service branches at|can (?:visit|use|access).{0,110}branch")
        for raw in page.markdown.splitlines():
            deadline = labeled_date(raw, r"\b(?:will close|closing on|scheduled to close(?: on)?)")
            if deadline:
                status = "was scheduled to close" if deadline <= cutoff else "is scheduled to close"
                return Answer(f"The branch {status} {deadline.strftime('%B')} {deadline.day}, {deadline.year}.",
                              cite(page, raw))
    return None


def branch_counts(version, cutoff):
    for page in articles(version, cutoff):
        match = re.search(r"\b(\d+)\s+branch (?:office )?locations\b", page.markdown, re.I)
        if match:
            yield match[1], page.source(match[0]), page.published


def branch_count(version, cutoff):
    latest = next(branch_counts(version, cutoff), None)
    if latest:
        value, source, _ = latest
        return Answer(f"Dupaco reports {value} branch office locations.", [source])
    return None


def service_conflicts(version):
    counts = {}
    for value, source, published in branch_counts(version, version.day):
        counts.setdefault(value, (source, published))
    if len(counts) < 2:
        return []
    winner = next(iter(counts))
    return [Conflict(
        topic="Dupaco branch office count",
        values=list(counts),
        sources=[source for source, _ in counts.values()],
        resolution=winner,
        reason=f"Used the latest dated publication that states the count ({counts[winner][1].isoformat()}).",
    )]


def eligibility(join, version, cutoff):
    if not join:
        return None
    lines = [line for line in join.markdown.splitlines() if re.search(
        r"^- (?:You live or work within|You have an immediate family member|You belong to)", line, re.I)]
    if not lines:
        return None
    text = " ".join(clean(line).lstrip("- ") for line in lines)
    sources = cite(join, *lines)
    for page in articles(version, cutoff):
        evidence = next((line for line in page.markdown.splitlines() if re.search(
            r"(?:serves (?:residents|members)|charter).{0,110}Iowa.{0,40}Illinois.{0,40}Wisconsin", line, re.I)), None)
        if evidence:
            text += " Dupaco's service area includes counties in Iowa, Illinois and Wisconsin."
            sources += cite(page, evidence)
            break
    return Answer(text, sources)


def answer_service(question, version, cutoff):
    query = words(question)
    docs = version.documents
    contact = docs.get("/contact")
    join = docs.get("/about/join")
    wants_link = bool(query & {"link", "url", "website"})
    if "routing" in query and ("number" in query or "aba" in query):
        primary = answer_line(contact, r"^Routing Number:\s*\d{9}\b")
        if primary:
            secondary = answer_line(docs.get("/business/cash-management"), r"routing number.{0,35}\d{9}\b")
            if secondary and re.search(r"\d{9}", primary.text)[0] in secondary.text:
                primary.sources.extend(secondary.sources)
            return primary
    if (wants_link and query & {"locator", "atm", "branch"}) or "locator" in query:
        return Answer(contact.url, cite(contact, "## Find a Location Near You")) if contact else None
    if wants_link and "certificate" in query and "rate" in query:
        page = docs.get("/personal/savings-account")
        return find_link(page, lambda label, url: label == "rates" and "#rates" in url) or (
            Answer(page.url + "#rates", cite(page, "## Savings Account Rates"))
            if page and "## Savings Account Rates" in page.markdown else None)
    if "join" in query and (wants_link or "online" in query or ("how" in question and "eligible" not in query)):
        return find_link(join, lambda label, url: ("join" in label or "become a member" in label)
                         and not url.rstrip("/").endswith("/login"))
    if "eligible" in query and query & {"join", "member"}:
        return eligibility(join, version, cutoff)
    if "branch" in query and ("how many" in question or "number of" in question):
        return branch_count(version, cutoff)
    if "branch" in query and query & {"close", "open", "full", "instead"}:
        closure = branch_closure(question, version, cutoff)
        if closure:
            return closure
    if ("lobby" in query or "drive" in query or "branch" in query) and query & {"hour", "hours", "schedule", "open", "close"}:
        return branch_hours(question, version, cutoff, contact)
    if "branch" in query and (query & {"address", "located"} or question.startswith("where")):
        return branch_address(question, contact)
    if "forbes" in query:
        for page in articles(version, cutoff, "forbes", "top-credit-union"):
            answer = answer_line(page, r"(?:Forbes.{0,35}(?:again|recogniz)|this was the \w+ time Forbes)")
            if answer:
                return answer
    if "volunteer time off" in question and "employee" in question:
        year = re.search(r"\b(?:19|20)\d\d\b", question)
        target = int(year[0]) if year else cutoff.year - 1 if "last year" in question else None
        for page in articles(version, cutoff, "volunteer"):
            answer = answer_line(page, r"\b[\d,]+\s+(?:Dupaco )?employees\b.{0,90}Volunteer Time Off")
            if not answer:
                continue
            reported = re.search(r"\bin ((?:19|20)\d\d)\b", answer.text)
            if target is None or (reported and int(reported[1]) == target):
                return answer
        if target is not None:
            return Answer("I can't confirm a Volunteer Time Off count for the requested year in the supplied records.",
                          confident=False)
    if re.search(r"\bhome(?:\$tart| ?start|tart)\b", question):
        for page in articles(version, cutoff, "hometart"):
            answer = answer_line(page, r"Home\$tart program.{0,120}provides up to")
            if answer:
                return answer
    wallet = query & {"wallet", "apple", "samsung", "google"} or "mobile payments" in question
    wallet_compatibility = re.search(
        r"^(?:can|could|may)\b.*\b(?:add|use)\b.{0,45}\bcards?\b.{0,20}\b(?:to|in|with)\b"
        r"|^(?:does|do|are|is)\b.*\b(?:cards?|dupaco)\b.{0,25}\b(?:work|support|compatible)\b"
        r"|\b(?:what|which)\s+(?:mobile |digital )?wallets?\b",
        question,
    )
    if wallet and wallet_compatibility and not re.search(
        r"\b(?:lost|lose|misplac\w*|stolen|missing|replac\w*|remove|disabl\w*|recover\w*)\b", question,
    ):
        pages = [doc for path, doc in docs.items() if path.endswith("/mobile-payments")]
        for page in pages:
            answer = answer_line(page, r"Apple Pay.*Google Pay.*Samsung Pay")
            if answer:
                return answer
    if (query & {"student", "college", "university"} and query & {"loan", "financing"}
            and not query & {"rate", "apr", "apy", "fee"}):
        page = docs.get("/personal/loans/student-loans")
        if page:
            flexibility = answer_line(page, r"Defer full payments|flexible (?:terms|repayment)")
            origination = answer_line(page, r"origination fees")
            if flexibility and origination:
                return Answer("Dupaco offers student loans for college and student loan refinancing. "
                              + flexibility.text + " " + origination.text,
                              cite(page, "# Student Loans & Refinancing")
                              + flexibility.sources + origination.sources)
    if ("update" in query and ({"contact", "information"} <= query or query & {"address", "email", "phone"})
            and not re.search(r"\b(?:alerts?|verification|codes?|two[ -]step|notifications?)\b", question)):
        if contact:
            phone = answer_line(contact, r"^Phone:\s*[\d-]+")
            if phone:
                return Answer("For help updating your contact details, call Dupaco at " +
                              phone.text.partition(":")[2].strip() + ".", phone.sources)
    return None
