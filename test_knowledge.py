"""Regression tests with invented evidence. Run with the kit's harness on PYTHONPATH."""
import unittest
from unittest.mock import patch

from contract import Page, Snapshot
from solution import build

BASE = "https://credit-union.example"


def page(path, markdown):
    return Page(BASE + path, path.rsplit("/", 1)[-1], markdown, "")


def snapshot(day, *pages):
    return Snapshot(day, day, list(pages))


def certificate(value):
    return page("/rates/certificates", f"# Certificates\n\n6-month certificate APY: {value}\n")


def auto_table(model_year, value):
    return (
        "# Auto Loans\n\n"
        "| Loan Program | Term | APR (As Low As) |\n"
        "| --- | --- | --- |\n"
        f"| {model_year} Model Year | 60 Months | {value} |\n"
    )


@patch.dict("os.environ", {"KNOWLEDGE_RAG": "0"})
class KnowledgeTests(unittest.TestCase):
    def test_history_removed_fee_and_expired_offer(self):
        kb = build()
        kb.ingest(snapshot(
            "2026-05-19",
            certificate("3.13%"),
            page("/fees", "# Fees\n\nLost debit card replacement fee: $4.25\n"),
        ))
        kb.ingest(snapshot(
            "2026-09-28",
            certificate("4.17%"),
            page("/promo", "# Offer\n\n6-month certificate APY: 3.13% (offer ended 2026-06-01)\n"),
        ))
        earlier = kb.answer("What was the APY on a six-month CD?", as_of="2026-05-31")
        self.assertIn("3.13%", earlier.text)
        self.assertEqual(earlier.sources[0].snapshot_id, "2026-05-19")
        latest = kb.answer("What is the APY on a six-month CD?")
        self.assertIn("4.17%", latest.text)
        self.assertNotIn("3.13%", latest.text)
        self.assertIn("$4.25", kb.answer("How much is a lost debit card replacement?", "2026-05-31").text)
        self.assertFalse(kb.answer("How much is a lost debit card replacement?").confident)
        self.assertFalse(kb.answer("What's the six-month certificate APY?", "2026-05-01").confident)

        comparison = kb.answer("How did the 6-month certificate APY change between May and September 2026?")
        self.assertIn("3.13%", comparison.text)
        self.assertIn("4.17%", comparison.text)
        self.assertEqual({source.snapshot_id for source in comparison.sources}, {"2026-05-19", "2026-09-28"})
        self.assertEqual([conflict.resolution for conflict in kb.conflicts()], ["4.17%"])

    def test_overlapping_model_year_conflict_and_unsupported_term(self):
        kb = build()
        kb.ingest(snapshot(
            "2026-09-28",
            page("/personal/loans/auto-motorcycle-loans", auto_table("2024-27", "5.21%")),
            page("/calculators/vehicle-affordability",
                 auto_table("2023-26", "5.02%") + "| 2023-26 Model Year | 48 Months | 4.55% |\n"),
        ))
        result = kb.answer("What APR can I get for five years on a 2025 car?")
        self.assertIn("5.21%", result.text)
        self.assertNotIn("5.02%", result.text)
        self.assertTrue(result.conflict_note)
        self.assertFalse(kb.answer("What's the APR on a 48-month auto loan for a 2025 vehicle?").confident)
        conflicts = kb.conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertIn("2024–2026", conflicts[0].topic)
        self.assertEqual(set(conflicts[0].values), {"5.21%", "5.02%"})
        self.assertEqual(conflicts[0].resolution, "5.21%")

    def test_distinct_metrics_and_balance_tiers(self):
        kb = build()
        kb.ingest(snapshot("2026-09-28", page("/personal/savings-account", """
# Savings Rates

Share Savings

| Account Type | Balance Tier | Dividend Rate | APY |
| --- | --- | --- | --- |
| Share Savings | $1 – $999.99 | 0.10% | 0.11% |
| Share Savings | $1,000+ | 0.20% | 0.21% |
""")))
        result = kb.answer("What is the Share Savings dividend rate on $1,250?")
        self.assertIn("0.20% dividend rate", result.text)
        self.assertNotIn("0.21%", result.text)
        self.assertNotIn("0.10%", result.text)
        self.assertEqual(kb.conflicts(), [])

    def test_business_tables_and_product_aliases(self):
        kb = build()
        header = """
| Account Type | Balance Tier | Dividend Rate | APY |
| --- | --- | --- | --- |
"""
        kb.ingest(snapshot(
            "2026-09-28",
            page("/personal/savings-account", "# Deposit rates\\n" + header +
                 "| Health Savings Account | $5,000+ | 1.20% | 1.25% |"),
            page("/business/savings", "# Business deposit rates\\n" + header +
                 "| Health Savings Account | $5,000+ | 2.20% | 2.25% |"),
        ))
        result = kb.answer("What annual percentage yield applies to $5k in my HSA?")
        self.assertIn("1.25%", result.text)
        self.assertNotIn("2.25%", result.text)
        self.assertEqual(result.sources[0].url, BASE + "/personal/savings-account")
        self.assertEqual(kb.conflicts(), [])

    def test_historical_conflict_uses_latest_evidence(self):
        kb = build()
        kb.ingest(snapshot(
            "2026-08-15",
            page("/personal/loans/auto-motorcycle-loans", auto_table("2024-27", "5.21%")),
            page("/calculators/vehicle-affordability", auto_table("2023-26", "5.02%")),
        ))
        kb.ingest(snapshot(
            "2026-09-28",
            page("/personal/loans/auto-motorcycle-loans", auto_table("2024-27", "5.44%")),
        ))
        disagreement = next(conflict for conflict in kb.conflicts() if "different values in" in conflict.reason)
        self.assertIn("2024–2026", disagreement.topic)
        self.assertEqual(disagreement.resolution, "5.44%")
        self.assertEqual(set(disagreement.values), {"5.02%", "5.21%", "5.44%"})
        self.assertTrue(any(source.snapshot_id == "2026-09-28" for source in disagreement.sources))
        self.assertTrue(any("changed between" in conflict.reason for conflict in kb.conflicts()))

    def test_fee_mirrors_and_faq_owner(self):
        kb = build()
        question = f"- [How do I deposit a check with the Shine app?]({BASE}/shine#faq)"
        instructions = "Endorse the check and select Mobile Deposit from Quick Actions on your phone."
        kb.ingest(snapshot(
            "2026-09-28",
            page("/fees", "# Fees\n\nIncoming wire transfer: $4.75"),
            page("/fee-schedule", "# Fees\n\n| Incoming wire transfer | $4.75 |"),
            page("/copied-faq", f"# Online banking\n\n{question}\n\n{instructions}"),
            page("/shine", f"# Online banking\n\n{question}\n\n{instructions}"),
        ))
        result = kb.answer("How much does it cost to receive a wire transfer?")
        self.assertIn("$4.75", result.text)
        result = kb.answer("Can I deposit a paper check from my phone?")
        self.assertIn("Mobile Deposit", result.text)
        self.assertEqual(result.sources[0].url, BASE + "/shine")

    def test_comparison_uses_the_day_before_shared_year(self):
        kb = build()
        for day, value in (("2026-05-12", "2.11%"), ("2026-05-19", "3.11%"), ("2026-06-09", "4.11%")):
            kb.ingest(snapshot(day, certificate(value)))
        answer = kb.answer("Compare the six-month certificate APY between May 12 and June 9, 2026.")
        self.assertEqual([source.snapshot_id for source in answer.sources], ["2026-05-12", "2026-06-09"])
        self.assertIn("2.11%", answer.text)
        self.assertNotIn("3.11%", answer.text)

    def test_certificate_regular_table_and_non_jumbo_scope(self):
        kb = build()
        kb.ingest(snapshot("2026-06-09", page("/rates", """# Rates

Special Certificates

| Term | Dividend Rate | APY | Min. to Earn |
| --- | --- | --- | --- |
| 25 Month Fixed Certificate Special | 3.57% | 3.60% | $1,000 |
| 25 Month Fixed Jumbo Certificate Special | 3.67% | 3.70% | $100,000 |

Regular

| Term | Dividend Rate | APY | Min. to Earn |
| --- | --- | --- | --- |
| 12 Month | 3.37% | 3.40% | $1,000 |
""")))
        self.assertIn("3.40%", kb.answer("What is the regular 12-month CD APY?").text)
        answer = kb.answer("What is the 25-month non-jumbo special CD APY?")
        self.assertIn("3.60%", answer.text)
        self.assertNotIn("3.70%", answer.text)

    def test_fee_comparison_uses_two_rows_and_keeps_conditions(self):
        kb = build()
        kb.ingest(snapshot("2026-06-09", page("/fees", """# Fees

Domestic wire $25.00 each
Incoming wire transfer $10.00
Stop payment (check, ACH, & bill pay) $30.00 per item
Per bill paid or attempted to pay
(Free if signed up for e-statements) $0.50
""")))
        answer = kb.answer("How much is the domestic wire fee, and does an incoming wire cost the same?")
        self.assertIn("$25.00", answer.text)
        self.assertIn("$10.00", answer.text)
        self.assertEqual(len(answer.sources), 2)
        bill = kb.answer("What fee applies if I pay a bill online without e-statements?")
        self.assertIn("$0.50", bill.text)
        self.assertIn("Free if signed up for e-statements", bill.text)
