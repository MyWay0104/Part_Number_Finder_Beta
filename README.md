# Part Number Finder Agent PoC

`Part Number Finder`는 CSV 또는 SQLite DB에 저장된 파트 카탈로그에서 사용자의 자연어 질문을 해석하고, 실제 데이터에 존재하는 `P\d{7}` 형식의 Part Number를 찾아주는 agentic PoC입니다.

이 프로젝트의 핵심 원칙은 LLM이 Part Number를 직접 만들지 않고, Python search tool이 반환한 후보 row만 해석한다는 점입니다.

## 주요 기능

- 한국어, 영어, 약어, 일부 오타/발음 표기 기반 Part Number 검색
- vendor/equipment 조건 필터링
- `Robot Blade`, `터보펌프`, `Quartz Tube`, `L/Q`, `W/Q`, `pmkit`, `T/V`, `오링` 같은 표현을 catalog part family로 정규화
- 문장 순서가 어색한 질문도 LLM/rule 기반으로 query plan 생성
- 두 개 이상의 파트가 들어간 질문을 파트별로 분해해 각각 검색
- 의미 기반 질문은 semantic catalog match와 TF-IDF row vector search로 후보 탐색
- candidate row 밖의 Part Number를 최종 답변에서 차단
- Langfuse tracing과 로컬 실패 로그(`data/search_failures.jsonl`) 지원

## Architecture

```text
User Input
  -> Agent Entry
     - main.py
     - answer_query_result()

  -> Query Understanding
     - llm_decompose_query(): multi-part QueryPlan 생성
     - llm_route(): single-route intent/tool 선택
     - rule_based_route(): LLM 비활성/실패 시 fallback

  -> LangGraph Multi-Part Workflow
     - decompose_query
     - search_each_part

  -> Search Tool Layer
     - exact / alias / fuzzy search
     - phonetic and abbreviation search
     - semantic catalog match
     - TF-IDF row vector search
     - vendor / equipment filter

  -> Candidate Validation
     - canonical part family 검증
     - vendor/equipment 조건 검증
     - confidence/status 판단

  -> Final Responder
     - tool-returned rows만 사용
     - 후보 밖 Part Number 차단
     - concise Korean response
```

## Core Files

```text
main.py
src/part_finder/agent.py
src/part_finder/llm_router.py
src/part_finder/search.py
src/part_finder/vector_index.py
src/part_finder/final_responder.py
src/part_finder/formatter.py
src/part_finder/data_loader.py
src/part_finder/normalizer.py
data/part_numbers.csv
data/aliases.json
data/semantic_hints.json
data/search_failures.jsonl
claude_ref.md
```

## Search Flow

1. `main.py`가 사용자 질문을 받고 `agent.py`의 `answer_query_result()`를 호출합니다.
2. `agent.py`는 먼저 `llm_decompose_query()`로 다중 파트 요청인지 확인합니다.
3. 다중 파트 요청이면 LangGraph workflow가 각 `PartLookupItem`을 별도로 검색합니다.
4. 단일 파트 요청이면 기존 `llm_route()` 기반 단일 검색 흐름으로 처리합니다.
5. `search.py`의 `agentic_part_search_tool()`이 exact, alias, phonetic, semantic, vector 후보를 병합합니다.
6. `agent.py`가 candidate row를 canonical part family와 vendor/equipment 조건으로 검증합니다.
7. `final_responder.py`가 tool 결과만 사용해 최종 답변을 생성합니다.

## Multi-Part Search

사용자 질문에 두 개 이상의 파트가 포함되면 `llm_decompose_query()`가 `QueryPlan`을 생성합니다.

예:

```bash
python main.py "AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘"
```

처리 방식:

- `Turbo Pump`와 `Robot Blade`를 별도 item으로 분리
- 공통 조건 `vendor=AMAT` 적용
- `Turbo Pump`는 AMAT row 반환
- `Robot Blade`는 AMAT 조건에서 없으면, 다른 vendor 후보를 별도로 안내

이 흐름은 한 파트가 실패해도 전체 답변을 실패 처리하지 않습니다.

## Canonical Part Family Validation

이번 업데이트에서 가장 중요한 변경점은 part family drift 방지입니다.

예를 들어 사용자가 `AMAT 로봇블레이드`를 요청했을 때, CSV에 AMAT `Robot Blade`가 없다고 해서 `Gate Valve`, `Cooling Plate`, `Shower Head` 같은 AMAT row를 반환하면 안 됩니다.

현재 agent는 다음 순서로 검증합니다.

1. 사용자 표현을 `Robot Blade` 같은 canonical part family로 정규화
2. 검색 후보가 해당 part family에 속하는지 확인
3. 그 다음 vendor/equipment 조건 적용
4. 조건에 맞는 row가 없으면 no result 또는 다른 조건 후보 안내

## Semantic / RAG Search

현재 구현은 외부 vector DB 없이 로컬 TF-IDF vector index를 사용합니다.

- chunk 단위: `part_numbers.csv`의 row 1개
- chunk 내용: `part_number`, `part_name`, `description`, `equipment_module`, `vendor_part_number`, `vendor`, aliases, semantic hints
- 검색 함수: `vector_semantic_search_tool`

예:

```bash
python main.py "ASML equipment robot arm pick part number"
```

정확한 파트명이 없어도 `robot arm pick`, `end effector`, `wafer transfer` 같은 의미 표현을 통해 `Robot Blade` 후보를 찾습니다.

## LLM Safety Policy

LLM은 Part Number를 직접 생성하지 않습니다.

- Part Number 후보는 CSV/DB/vector search 결과에서만 옵니다.
- final responder는 `candidate_rows`에 포함된 Part Number만 사용할 수 있습니다.
- LLM 답변에 후보 밖 Part Number가 있으면 fallback 답변으로 대체합니다.
- vendor/equipment는 part name 후보가 아니라 필터로 취급합니다.
- multi-part query는 item별로 성공/실패를 분리합니다.

## Environment

`.env` 예시:

```env
PART_FINDER_USE_LLM=1
OLLAMA_MODEL=gemma3:4b
OLLAMA_HOST=http://localhost:11434
```

LLM을 끄고 deterministic fallback만 사용하려면:

```env
PART_FINDER_USE_LLM=0
```

Langfuse tracing은 선택 사항입니다.

```env
PART_FINDER_TRACE_LANGFUSE=1
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://your-langfuse-server
LANGFUSE_PROJECT=part-number-finder
```

## Company TXT to CSV / RAG Workflow

사내 배포에서는 원천 TXT 데이터를 먼저 canonical CSV schema로 변환하는 것을 권장합니다.

```bash
python scripts/preprocess_txt_to_csv.py company_parts.txt --output data/part_numbers.csv
```

필수 schema:

```csv
part_number,part_name,description,equipment_module,vendor_part_number,vendor
```

row chunk RAG index 생성:

```bash
python scripts/build_vector_index.py data/part_numbers.csv --output data/part_vector_index.json
```

사내 embedding model이 있으면 `list[str] -> list[list[float]]` callable로 노출한 뒤 사용할 수 있습니다.

```bash
python scripts/build_vector_index.py data/part_numbers.csv --embedding-provider company_embeddings:embed_texts
```

내장 TF-IDF 검색은 fallback으로 계속 동작합니다.

## Run

설치:

```bash
uv sync
```

단일 질문:

```bash
python main.py "W/Q part number 알려줘"
python main.py "Owe ling part number"
python main.py "Lam Research 장비 Vacuum Gauge part number 알려줘"
python main.py "ASML equipment robot arm pick part number"
```

다중 파트 질문:

```bash
python main.py "AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘"
python main.py "파트넘버 알려줘 W/Q 그리고 L/Q AMAT"
```

interactive mode:

```bash
python main.py
```

종료:

```text
q
quit
exit
```

## Data Schema

권장 CSV schema:

```csv
part_number,part_name,description,equipment_module,vendor_part_number,vendor
P2100958,PM Kit,PM Kit for Helios XP,Helios XP,AMAT-PMK-2100958,AMAT
```

필수/권장 컬럼:

- `part_number`: 필수, `P\d{7}` 형식
- `part_name` 또는 `description`: 둘 중 하나 필수
- `equipment_module`: 선택
- `vendor_part_number`: 선택
- `vendor`: 선택

데이터 로딩 우선순위:

1. 명시적으로 전달한 path
2. 프로젝트 루트의 `Part_Number.db`
3. `data/part_numbers.csv`
4. 프로젝트 루트의 `*.csv`
5. 프로젝트 루트의 `*.txt`

## Alias And Semantic Hints

사용자 표현과 실제 catalog name을 연결하려면 `data/aliases.json`을 수정합니다.

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz"],
  "O-ring": ["o ring", "oring", "owe ling"]
}
```

기능/문맥 기반 검색을 강화하려면 `data/semantic_hints.json`을 보강합니다.

```json
{
  "Robot Blade": ["robot arm", "wafer transfer", "pick", "end effector"]
}
```

## Internal Claude Code Handoff

`claude_ref.md`는 사내 Claude Code 환경에 전달하기 위한 참고 문서입니다.

포함 내용:

- 현재 코드 구조
- Ollama에서 API-key 기반 `ChatOpenAI` 방식으로 전환할 때 검토할 파일
- TXT -> CSV -> RAG pipeline 검토 사항
- `.env` 변수값
- 사내 환경에서 유지해야 할 safety rule

민감정보가 포함될 수 있으므로 외부 공유에 주의해야 합니다.

## Tests

전체 테스트:

```bash
uv run pytest
```

현재 검증 범위:

- dummy data 생성과 Part Number 형식 검증
- alias normalization
- exact/fuzzy search
- vendor/equipment filter
- semantic catalog match
- CSV row chunk vector semantic search
- agent confirmation flow
- multi-part LangGraph workflow
- canonical part family drift 방지
- final responder hallucination guard
- Langfuse/config behavior

마지막 확인 결과:

```text
45 passed
```

`.pytest_cache` 쓰기 권한 경고가 표시될 수 있지만, 테스트 자체가 통과하면 기능 검증에는 영향이 없습니다.
