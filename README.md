# Multi-Model Cost and Quality Router for RAG

This project implements a Retrieval-Augmented Generation (RAG) system that routes each query to the cheapest model likely to answer correctly.

It combines:

- LangChain for ingestion, chunking, embeddings, retrieval, and chain orchestration.
- LiteLLM for multi-provider model abstraction, fallback, and cost accounting.
- A query complexity classifier for tiered routing.
- A confidence-triggered escalation path to stronger models.
- A semantic cache that returns answers for near-duplicate queries at zero model cost.
- Evaluation scripts and a Streamlit dashboard for cost-vs-accuracy analysis.
- Threshold sweep experiments to map confidence policy to cost/quality outcomes.

## Project Structure

```
project 1/
  data/
    corpus/                # Source docs used by RAG
    eval/eval_set.csv      # Ground-truth evaluation set
  dashboard/app.py         # Streamlit dashboard
  docs/
    IMPLEMENTATION_GUIDE.md
  logs/                    # Query logs and eval outputs
  src/
    cache/
      semantic_cache.py
    config.py
    main.py
    rag/
      llm_client.py
      schemas.py
      vector_store.py
    router/
      classifier.py
      confidence.py
      model_router.py
    pipeline/
      rag_pipeline.py
    evaluation/
      evaluator.py
      threshold_sweep.py
    utils/logger.py
  scripts/
    run_all.ps1
    run_all.bat
  requirements.txt
  .env.example
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment template and add API keys:

   ```bash
   copy .env.example .env
   ```

4. Ingest corpus into Chroma:

   ```bash
   python -m src.main ingest
   ```

5. Ask a query through routed RAG:

   ```bash
   python -m src.main ask --query "How should I retry charge creation after a timeout?"
   ```

6. Run evaluation scenarios:

   ```bash
   python -m src.main eval
   ```

7. Run threshold sweep experiment:

  ```bash
  python -m src.main sweep --start-threshold 0.60 --end-threshold 0.90 --step 0.05
  ```

8. One-click pipeline run (ingest + eval + sweep):

  ```powershell
  scripts\run_all.ps1
  ```

9. Launch dashboard:

   ```bash
   streamlit run dashboard/app.py
   ```

10. Optional Windows one-click launcher:

  ```bat
  scripts\run_all.bat
  ```

11. Run with a specific domain preset:

  ```bash
  python -m src.main run-all --domain support
  python -m src.main run-all --domain devdocs
  python -m src.main run-all --domain compliance
  ```

## Routing Logic

1. Classify query as simple, complex, or ambiguous.
2. Route to model tier:
   - simple -> tier_1 (cheap/fast)
   - complex/ambiguous -> tier_2 (stronger)
3. Generate answer with retrieved context.
4. Compute confidence (model self-score + heuristic blend).
5. If confidence is below threshold, escalate to tier_3.
6. Log query, tier, model, confidence, latency, and cost.

## Domain Presets

The app supports three runnable routing profiles:

- `support`
- `devdocs`
- `compliance`

How to use:

- `python -m src.main ask --domain support --query "..."`
- `python -m src.main eval --domain devdocs`
- `python -m src.main sweep --domain compliance --start-threshold 0.70 --end-threshold 0.90 --step 0.05`
- `python -m src.main run-all --domain support`

Default behavior:

- If `--domain` is omitted, the app uses `ROUTER_DOMAIN` from `.env`.
- If `ROUTER_DOMAIN` is not set, it defaults to `support`.

Preset policy summary:

- `support`: favors tier_1 for simple queries, forces tier_3 for high-risk support intents.
- `devdocs`: routes direct API lookups to tier_1, how-to to tier_2, debugging/migration queries to tier_3.
- `compliance`: conservative routing, generally tier_2 or tier_3 with stricter confidence and source checks.

## Evaluation Scenarios

- cheap_only: all queries forced to tier_1
- expensive_only: all queries forced to tier_3
- router: classifier + routing + escalation

Outputs:

- `logs/eval_results.csv` for per-query outcomes
- `logs/eval_summary.json` for scenario aggregates
- `logs/query_log.jsonl` for full query traces
- `logs/semantic_cache.json` for persistent semantic cache entries
- `logs/threshold_sweep.csv` for threshold experiment runs
- `logs/threshold_sweep_summary.json` for best-threshold summary

Domain-aware output fields include:

- `domain`
- `route_reason`
- `source_count`

## Semantic Cache

- Cache is enabled by default for interactive queries.
- Evaluation disables cache by default for fair scenario comparisons.
- Cache hit returns cached answer with:
  - `cost_usd = 0`
  - `cache_hit = true`
  - `cache_similarity` as nearest-neighbor score

Tune cache policy in `.env`:

- `ENABLE_SEMANTIC_CACHE`
- `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`
- `SEMANTIC_CACHE_MAX_ENTRIES`
- `SEMANTIC_CACHE_MIN_CONFIDENCE`

## Dashboard Domain KPIs

The Streamlit dashboard includes a domain selector and a `Domain KPI Targets` panel.

Per-domain targets are checked against live metrics and shown as `pass` or `watch` for:

- Router accuracy
- Cost savings vs expensive baseline
- Tier 1 p95 latency
- Escalation rate
- Cache hit rate
- Multi-source rate (compliance preset)

## Detailed Documentation

See `docs/IMPLEMENTATION_GUIDE.md` for architecture, runbook, design decisions, and interview talking points.

For full project-level technical reference (all parameters, modules, schemas, and operational notes), see `docs/COMPLETE_PROJECT_DOCUMENTATION.md`.
