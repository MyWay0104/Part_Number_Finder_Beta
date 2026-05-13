# Claude Code Handoff Reference

This file is intended for the internal Claude Code environment after cloning this repository. Treat it as sensitive because it includes environment variable values that are normally excluded by `.gitignore`.

## Current Runtime Snapshot

- Entry point: `main.py`
- Main public API: `part_finder.agent.answer_query_result(query, top_k=3, pending_confirmation=None)`
- Current LLM runtime: local Ollama through `langchain_ollama.ChatOllama`
- Current default model: `gemma3:4b`
- Current agent workflow:
  1. `agent.py` receives the user query.
  2. Multi-part queries are sent through a LangGraph workflow in `_run_multi_part_workflow`.
  3. `llm_router.py` decomposes the query into `QueryPlan` and `PartLookupItem` objects.
  4. Each item is converted into a `RouteDecision`.
  5. `search.py` runs exact/alias/phonetic/semantic/vector search through `agentic_part_search_tool`.
  6. `agent.py` validates candidates against vendor/equipment filters and canonical part family.
  7. `final_responder.py` formats final answers using only tool-returned rows.

## Important Recent Behavior

- Multi-part queries are now supported.
  - Example: `AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘`
  - The workflow searches `Turbo Pump` and `Robot Blade` separately.
- Canonical part family validation was tightened.
  - If the user asks for `Robot Blade`, the agent must not return `Gate Valve`, `Cooling Plate`, etc. just because those rows match the vendor.
- If a vendor/equipment filter eliminates a requested part, the multi-part workflow relaxes the context only for alternatives.
  - Example: AMAT `Robot Blade` is not found in the CSV, but ASML/TEL `Robot Blade` alternatives may be shown.
- `main.py` no longer prompts for retry input when the query was passed as a command-line argument.

## Files To Read First

1. `README.md`
2. `main.py`
3. `src/part_finder/agent.py`
4. `src/part_finder/llm_router.py`
5. `src/part_finder/search.py`
6. `src/part_finder/vector_index.py`
7. `src/part_finder/final_responder.py`
8. `src/part_finder/data_loader.py`
9. `tests/test_agent.py`
10. `tests/test_agentic_semantic.py`
11. `tests/test_rag_index.py`

## Environment Variables From Local `.env`

```env
PART_FINDER_USE_LLM=1
OLLAMA_MODEL=gemma3:4b
OLLAMA_HOST=http://localhost:11434

PART_FINDER_TRACE_LANGFUSE=1
LANGFUSE_PUBLIC_KEY=pk-lf-9b4b874f-ce3e-43da-b439-a0bfc2944446
LANGFUSE_SECRET_KEY=sk-lf-64af3bcc-df99-46f7-97f5-65a80b607ec6
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGFUSE_PROJECT=Default_Tracing

PYTHONPATH=src
```

## LLM Runtime Migration Notes

The current code uses Ollama directly in two places:

- `src/part_finder/llm_router.py`
  - `llm_route`
  - `llm_decompose_query`
- `src/part_finder/final_responder.py`
  - `_call_final_llm`

Internal deployment is expected to use API-key based models such as GLM-5.1 or gemma4, likely through an OpenAI-compatible endpoint and `langchain_openai.ChatOpenAI`.

Claude should review and change:

- `src/part_finder/config.py`
  - Add API model settings such as `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, or internal equivalents.
  - Keep backward compatibility if local Ollama remains useful for development.
- `src/part_finder/llm_router.py`
  - Replace or wrap `ChatOllama` construction with a provider factory.
  - Keep `temperature=0` for routing/decomposition.
  - Preserve JSON-only prompts and strict schema parsing.
- `src/part_finder/final_responder.py`
  - Use the same provider factory.
  - Keep hallucination guards: final text may only mention part numbers present in `candidate_rows`.

Do not allow the LLM to create final part numbers. It should only choose tools, normalize terms, decompose user intent, and summarize tool-returned candidates.

## Database And RAG Pipeline Notes

The internal source database is expected to be a TXT file that must be transformed into CSV before search/RAG.

Current repo components:

- `scripts/preprocess_txt_to_csv.py`
- `data/part_numbers.csv`
- `src/part_finder/data_loader.py`
- `src/part_finder/vector_index.py`
- `scripts/build_vector_index.py`
- `data/aliases.json`
- `data/semantic_hints.json`

The CSV schema expected by the app:

```csv
part_number,part_name,description,equipment_module,vendor_part_number,vendor
```

Required fields:

- `part_number`: must match `P\d{7}` in the current PoC.
- `part_name` or `description`: at least one must exist.

Recommended internal ingestion steps:

1. Parse TXT into normalized structured rows.
2. Normalize headers into the canonical schema in `data_loader.FIELD_ALIASES`.
3. Build or update `data/part_numbers.csv`.
4. Expand `data/aliases.json` from real user vocabulary, Korean terms, abbreviations, typo patterns, and vendor naming conventions.
5. Expand `data/semantic_hints.json` by part family.
6. Rebuild any vector index artifact if persistent vector search is introduced.

## Vector Index Review Points

`src/part_finder/vector_index.py` currently uses an in-memory TF-IDF row-level index. This is simple and deterministic, but it is not a production-grade embedding pipeline.

Current strengths:

- No external service required.
- Works in restricted environments.
- One CSV row equals one RAG chunk, which avoids mixing unrelated part numbers.
- Includes aliases and semantic hints in each row chunk.

Likely internal improvements:

- Replace or supplement TF-IDF with the internal embedding model/API.
- Add a persistent vector store if the real catalog is large.
  - Candidate options: FAISS, Chroma, SQLite vector extension, or an internal vector DB.
- Keep row-level chunks for final PN safety.
- Consider additional chunk types:
  - part family summary chunks
  - alias/synonym chunks
  - equipment-specific chunks
  - vendor-specific chunks
- Add reranking that respects exact part family first, then vendor/equipment filters.
- Keep deterministic validation after vector retrieval so semantically similar but wrong part families do not pass.

## Agentic Workflow Notes

The current LangGraph workflow lives in `src/part_finder/agent.py`:

- `_run_multi_part_workflow`
- nodes:
  - `decompose_query`
  - `search_each_part`

It uses `llm_decompose_query` from `llm_router.py`, then searches every `PartLookupItem`. If the query contains only one item, it falls back to the existing single-route flow.

Claude can improve this by splitting the workflow into explicit modules if production complexity grows:

- `planner.py`: query decomposition and schema validation
- `resolver.py`: canonical part family resolution
- `workflow.py`: LangGraph graph construction
- `validators.py`: final candidate validation

Keep the current public API stable unless the application layer is also updated.

## Behavioral Tests To Preserve

Run:

```powershell
uv run pytest
```

Important cases:

- `로봇블레이드 파트넘버. AMAT 장비 기준으로.`
  - Must not return `Gate Valve` or `Cooling Plate`.
  - If AMAT has no Robot Blade row, no AMAT result should be returned.
- `AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘`
  - Must search both parts.
  - Must return AMAT Turbo Pump.
  - Must separately report Robot Blade as missing under AMAT, with non-AMAT alternatives if available.
- `파트넘버 알려줘 W/Q 그리고 L/Q AMAT`
  - Must search W/Q and L/Q separately.
- `ASML equipment robot arm pick part number`
  - Semantic query should still produce Robot Blade candidates and use confirmation/candidate behavior.

## Safety Rules To Keep

- Tool-returned rows are the only source of part numbers.
- The final LLM response must not introduce a `P\d{7}` value outside candidate rows.
- Vendor and equipment terms are filters, not part-name candidates.
- Canonical part family matching must happen before accepting fuzzy matches.
- Multi-part queries should degrade per item, not fail the entire answer because one item has no match.
