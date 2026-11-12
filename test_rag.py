"""Exercise RAG through the evaluator entry point with synthetic snapshots."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from contract import Answer, Page, Snapshot
from rag import DIMENSIONS, OpenAI, OpenAIError, Rag, numbered_excerpts
from evidence import Document, Passage, TextIndex, markdown_blocks
from solution import Knowledge


class RagTests(unittest.TestCase):
    def test_policy_questions_do_not_get_intercepted_by_topic_keywords(self):
        day = "2026-09-28"
        wallet = Page("https://example.org/online-and-mobile-banking/mobile-payments", "Wallets",
                      "Add your cards to Apple Pay, Google Pay or Samsung Pay.", day)
        contact = Page("https://example.org/contact", "Contact", "Phone: 800-555-0100", day)
        rag = Mock()
        rag.answer.return_value = Answer("A prose answer.")
        knowledge = Knowledge(rag)
        knowledge.ingest(Snapshot(day, day, [wallet, contact]))
        questions = (
            "What support number should I call about Shine login?",
            "How do I remove my phone number from someone else's alerts?",
            "Can I change the email used for verification codes?",
            "I lost my phone with my card in its mobile wallet. What should I do?",
            "Does Apple Pay work when the phone is offline?",
            "Does Dupaco charge a fee to use Apple Pay?",
            "Do Bright Savings buckets earn differently?",
            "What must the Bright Savings bucket percentages add up to?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(knowledge.answer(question).text, "A prose answer.")
                self.assertEqual(rag.answer.call_args.args[0], question)
        self.assertIn("Samsung Pay", knowledge.answer("Can I add my debit card to a mobile wallet?").text)

    def test_semantic_answer_uses_requested_snapshot_and_real_source(self):
        old = "Library members can renew books twice by calling the lending desk."
        new = "Library members can renew books three times using the member portal."
        question = "How can I extend the time I have an item?"
        requested_date = "2026-06-01"
        latest_date = "2026-08-01"
        url = "https://example.org/renewals"
        client = OpenAI("test-key", "test-decisions", "test-responses")
        calls = []

        def post(endpoint, payload, deadline):
            calls.append(endpoint)
            if endpoint == "embeddings":
                vector = [1.0] + [0.0] * (DIMENSIONS - 1)
                return {"data": [
                    {"index": index, "embedding": vector}
                    for index in range(len(payload["input"]))
                ]}
            if endpoint == "decisions":
                self.assertIn(old, payload["input"])
                self.assertNotIn(new, payload["input"])
                return {"answers": [{"name": "p1", "type": "predicate", "probability": 0.98}]}
            evidence = json.loads(payload["input"])["passages"]
            self.assertEqual(evidence[0]["snapshot_date"], requested_date)
            self.assertEqual(evidence[0]["url"], url)
            self.assertIn("authority", evidence[0])
            return {"status": "completed", "output": [{"type": "message", "content": [{
                "type": "output_text",
                "text": json.dumps({"answer": "Call the lending desk to renew books. [Citations are listed separately.]",
                                    "status": "answered",
                                    "citations": ["p1.1"]}),
            }]}]}

        with TemporaryDirectory() as directory, patch.object(client, "post", side_effect=post):
            knowledge = Knowledge(Rag(client, Path(directory) / "index.sqlite3", use_decisions=True))
            for day, passage in ((requested_date, old), (latest_date, new)):
                knowledge.ingest(Snapshot(day, day, [Page(url, "Renewals", passage, day)]))
            answer = knowledge.answer(question, as_of=requested_date)

        self.assertEqual(calls, ["embeddings", "embeddings", "embeddings", "decisions", "responses"])
        self.assertTrue(answer.confident)
        self.assertEqual(answer.text, "Call the lending desk to renew books.")
        self.assertEqual(answer.sources[0].snapshot_id, requested_date)
        self.assertEqual(answer.sources[0].url, url)
        self.assertEqual(answer.sources[0].quote, old)

    def test_rejects_unselected_citations_and_preserves_local_fallback(self):
        passage = "Use the library website to renew borrowed books online."
        client = OpenAI("test-key", "test-decisions", "test-responses")
        with TemporaryDirectory() as directory:
            knowledge = Knowledge(Rag(client, Path(directory) / "index.sqlite3", use_decisions=True))
            day = "2026-08-01"
            page = Page("https://example.org/renewals", "Renew borrowed books", passage, day)
            unrelated = [
                Page(f"https://example.org/hours/{index}", "Opening hours",
                     f"The reading room opens at noon on the {index}th day of the month.", day)
                for index in range(1, 12)
            ]
            with patch.object(client, "embed", side_effect=OpenAIError("unavailable")), self.assertLogs("rag"):
                knowledge.ingest(Snapshot(day, day, [page, *unrelated]))
            with (
                patch.object(client, "select", return_value=["p1"]),
                patch.object(client, "generate", return_value={
                    "answer": "Call this number to renew.", "status": "answered", "citations": ["p1.99"],
                }),
            ):
                answer = knowledge.answer("How do I renew borrowed books?")
            self.assertEqual(answer.text, passage)
            with (
                patch.object(client, "select", return_value=["p1"]),
                patch.object(client, "generate", return_value={
                    "answer": "Call the lending desk at 100.",
                    "status": "answered", "citations": ["p1.1"],
                }),
            ):
                answer = knowledge.answer("How do I renew borrowed books?")
            self.assertEqual(answer.text, passage)
            with patch.object(client, "select", return_value=[]):
                answer = knowledge.answer("How do I renew borrowed books?")
            self.assertFalse(answer.confident)

    def test_future_effective_policy_does_not_replace_available_prose(self):
        day = "2026-05-12"
        page = Page("https://example.org/digital-privacy-policy", "Policy",
                    "# Privacy Policy\n\nEffective as of: May 19, 2026\n\n"
                    "A shared user may download the privacy report from the profile screen.", day)
        knowledge = Knowledge()
        knowledge.ingest(Snapshot(day, day, [page]))
        answer = knowledge.answer("Where can a shared user download the privacy report?")
        self.assertFalse(answer.confident)

    def test_faq_list_and_policy_table_keep_the_original_context(self):
        markdown = """# Login

- [Which domains should I allow?](https://example.org/help#)

Allow these domains:

- login.example.org
- verify.example.org

## Account privileges

| Operation | Primary | Joint |
| --- | --- | --- |
| Add a loan | Yes | No |
"""
        document = Document.from_page(Page("https://example.org/help", "Help", markdown, ""), "test")
        passages = [block for block in markdown_blocks(document) if isinstance(block, Passage)]
        faq = next(block for block in passages if block.question)
        self.assertIn("Allow these domains", faq.text)
        self.assertIn("verify.example.org", faq.text)
        table = next(block for block in passages if "Add a loan" in block.text)
        self.assertIn("Primary | Joint", table.text)
        self.assertIn(table.raw, markdown)

    def test_platform_label_stays_with_its_own_procedure(self):
        markdown = """# Manage your alerts

## How to receive two-step verification codes

**In Shine Mobile:**

From Profile, select “Two Step Notification”, then “Add contact”.

**In Shine Online:**

From Profile, select “Edit Profile”, open “Actions”, and select “edit two-step”.
"""
        page = Page("https://example.org/alerts", "Alerts", markdown, "")
        passages = list(markdown_blocks(Document.from_page(page, "test")))
        index = TextIndex(passages)
        for channel, platform, own_label, other_label in (
            ("the browser", "web", "edit two-step", "Add contact"),
            ("the Shine app", "mobile", "Add contact", "edit two-step"),
        ):
            with self.subTest(channel=channel):
                passage = index.search(f"Where do I change my sign-in codes in {channel}?")[0][0]
                self.assertEqual(passage.platform, platform)
                self.assertIn(own_label, passage.text)
                self.assertNotIn(other_label, passage.text)

    def test_unicode_soft_break_preserves_the_complete_card_instruction(self):
        raw = "No physical card yet? Go to your account with your \u2028card and tap “View card details”."
        page = Page("https://example.org/cards", "Cards", raw, "")
        passage, = markdown_blocks(Document.from_page(page, "test"))
        self.assertEqual(passage.raw, raw)
        self.assertIn("Go to your account with your card", passage.text)
        self.assertIn("Go to your account with your card and tap", numbered_excerpts("p1", passage)["p1.2"])
