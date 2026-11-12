# Knowledge system for dated website snapshots

## Run it

Use Python 3.11. The code uses the standard library, and the participant kit provides the harness, data, and public questions. From the unpacked kit, replace `/path/to/knowledge-hackathon` with the path to this clone.

```sh
python -m pip install -r /path/to/knowledge-hackathon/requirements.txt
KNOWLEDGE_RAG=0 python harness/run_eval.py \
  --submission /path/to/knowledge-hackathon --eval eval/public_eval.jsonl
PYTHONPATH="$PWD/harness" KNOWLEDGE_RAG=0 python -m unittest discover \
  -s /path/to/knowledge-hackathon
```

## How the system decides what's current

Each crawl has its own index. An undated question uses the newest crawl. An as-of question uses the last crawl on or before the requested date. A rate comparison resolves each date independently and cites the selected snapshot for each side. A page's capture date tells us when it was observed. Published effective dates and offer end dates determine whether a fact applies to the requested date.

The local path handles rates, fees, and specific service questions. It parses tables into structured observations and preserves their headers, qualifiers, and source excerpts. Before comparing values, it matches the product, APR or APY, term, account type, and applicable balance or vehicle tier. A fee comparison can select multiple rows and keep conditions such as free-use allowances. For rates, an official rate schedule or product page takes precedence over a calculator or promotion. For sources of similar authority, the resolver prefers the later applicable effective date. If equally applicable rates still disagree, it abstains. The conflict report covers structured rate conflicts and documented branch counts. It has limited coverage of contradictions in free text.

## Approaches tried

- **Local rules and keyword search.** Structured lookup gave fast, reproducible answers for dates, rates, and fees. Keyword search alone struggled when members used different words from the site. We narrowed the rules so a word like “number” or “earn” does not intercept a support or product-policy question.
- **Hybrid retrieval for prose.** Keyword and vector search each produce up to 24 unique passages from the chosen snapshot. Their rankings are combined into a shortlist of up to 12. Each passage keeps its heading and source. The parser groups FAQ answers with their lists and attaches browser or app labels to the relevant procedure. A policy table keeps its column labels. This helps retrieve a complete instruction instead of just an introductory paragraph or the last menu label.
- **Separate evidence selection versus direct answering.** We tested a GPT-6 Luna Decisions call before answer generation, then tested sending the same shortlist directly to the answer model. With GPT-5.4 nano, Decisions tied on both the 45-question custom set and the eight-question follow-up in that comparison. It added 0.25 seconds to p95 on the 45. Direct answering became the default, and Decisions is available as an opt-in experiment.
- **Copied quotes versus excerpt IDs.** Asking the model to copy its quotes made the response longer and led to frequent fallbacks with Nano. The final prompt numbers the excerpts and asks for their IDs. The code verifies that each ID exists, that numeric details occur in the cited excerpts, and that returned citations point to the original URL and snapshot. The model still judges whether the evidence supports each prose claim. It can report partial support or abstain. Nano with excerpt IDs performed better than Luna in the end-to-end comparison.

## API key and data

**The submission runs without an API key.** The keyless path uses structured lookup and local text search. To enable the optional paid OpenAI path, set `OPENAI_API_KEY` in the process environment or put it in a local `.env` beside `solution.py`. The same key is used for `text-embedding-3-small` to embed passages and questions, and `gpt-5.4-nano` to answer prose questions. Set `KNOWLEDGE_DECISIONS=1` to add the optional GPT-6 Luna Decisions call. Use `KNOWLEDGE_RAG=0` to force keyless mode even if a key is present. API failures fall back to local search.

The optional API path sends snapshot passages for embedding and sends the question and retrieved evidence for answering. It makes no live requests to the credit union website. The SQLite embedding cache can be rebuilt locally from the kit. Keep the snapshots, API key, and cache out of the repository.

## Verification

A fresh Python 3.11 clone scored **94.6%** on the 40 public questions in keyless mode, with zero stale-value hits and 20 consistent repeats. All **16 tests passed**. With the optional API enabled, the final 45-question custom run scored **91.5%** at **2.92 seconds p95** with zero stale-value hits. Both previously missed Shine procedures and four fresh paraphrases earned full credit after the procedure parsing fix. These results cover the available evaluation questions. The hidden evaluation is separate.
