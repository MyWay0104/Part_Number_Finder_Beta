# Part Number Finder Agent User Guide

이 문서는 현재 구현된 Part Number Finder Agent PoC의 사용 방법과 내부 동작 방식을 설명합니다.

## 현재 상태

현재 프로젝트는 단순 정규식/alias 기반 finder가 아니라, LLM과 Python tool을 함께 사용하는 agentic finder 구조입니다.

중요한 설계 원칙:

- LLM은 사용자의 질문 의도, 파트명 후보, 검색 전략을 판단합니다.
- 실제 Part Number 후보는 Python search tool, CSV/DB, vector index 결과에서만 가져옵니다.
- LLM은 Part Number를 새로 만들 수 없습니다.
- 두 개 이상의 파트가 질문에 포함되면 파트별로 분리 검색합니다.
- vendor/equipment 조건 때문에 다른 part family로 잘못 drift 되는 것을 validation 단계에서 차단합니다.
- LLM이 실패하거나 비활성화되어도 deterministic fallback으로 동작합니다.

## 전체 구성

```text
.
├── main.py
├── README.md
├── User_Guide.md
├── claude_ref.md
├── pyproject.toml
├── data
│   ├── aliases.json
│   ├── semantic_hints.json
│   ├── part_numbers.csv
│   └── search_failures.jsonl
├── scripts
│   ├── generate_dummy_data.py
│   ├── preprocess_txt_to_csv.py
│   └── build_vector_index.py
├── src
│   └── part_finder
│       ├── agent.py
│       ├── config.py
│       ├── data_loader.py
│       ├── final_responder.py
│       ├── formatter.py
│       ├── llm_router.py
│       ├── normalizer.py
│       ├── rag_index.py
│       ├── search.py
│       ├── tracing.py
│       └── vector_index.py
└── tests
    ├── test_agent.py
    ├── test_agentic_semantic.py
    ├── test_config.py
    ├── test_dummy_data.py
    ├── test_formatter.py
    ├── test_langfuse_tracing.py
    ├── test_normalizer.py
    ├── test_rag_index.py
    └── test_search.py
```

## 실행 준비

Python 3.13 이상 기준입니다.

```bash
uv sync
```

일반 Python 환경에서는 `pyproject.toml`의 dependencies를 설치한 뒤 실행하면 됩니다.

## 기본 실행

단일 질문:

```bash
python main.py "W/Q part number 알려줘"
```

반환 후보 수 조정:

```bash
python main.py --top-k 5 "PM Kit part number 알려줘"
```

대화형 모드:

```bash
python main.py
```

종료:

```text
q
quit
exit
```

명령행 인자로 질문을 넘긴 경우에는 결과 없음 상황에서도 추가 입력 prompt를 띄우지 않습니다. 대화형 모드에서만 retry/confirmation 흐름이 이어집니다.

## LLM 설정

기본 구조는 Ollama 기반 LLM router와 final responder를 사용합니다.

`.env` 예시:

```env
PART_FINDER_USE_LLM=1
OLLAMA_MODEL=gemma3:4b
OLLAMA_HOST=http://localhost:11434
```

LLM을 끄고 deterministic rule-only 모드로 실행하려면:

```env
PART_FINDER_USE_LLM=0
```

`OLLAMA_HOST`가 `http://localhost:11434/v1`처럼 OpenAI-compatible path를 포함해도 `config.py`에서 `http://localhost:11434`로 정규화합니다.

## Langfuse Tracing

Langfuse tracing은 선택 사항입니다.

```env
PART_FINDER_TRACE_LANGFUSE=1
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://your-langfuse-server
LANGFUSE_PROJECT=part-number-finder
```

로컬 테스트에서 tracing 전송을 끄려면:

```env
PART_FINDER_TRACE_LANGFUSE=0
```

Langfuse와 별개로 `data/search_failures.jsonl`에는 실패/모호 후보 로그가 남습니다. 이 로그는 `aliases.json`과 `semantic_hints.json` 튜닝에 사용합니다.

## 내부 동작

### 1. 사용자 입력 수신

`main.py`가 사용자 질문을 받고 `agent.py`의 `answer_query_result()`를 호출합니다.

### 2. Query Decomposition

`llm_router.py`의 `llm_decompose_query()`가 사용자 질문을 `QueryPlan`으로 변환합니다.

`QueryPlan`에는 다음 정보가 들어갑니다.

- 검색해야 할 파트 목록: `PartLookupItem`
- 요청 필드: 예를 들어 `part_number`, `vendor`, `equipment_module`
- 공통 vendor 조건
- 공통 equipment 조건
- LLM 사용 여부

예:

```text
AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘
```

분해 결과 개념:

```text
vendor_query = AMAT
items = [Turbo Pump, Robot Blade]
requested_fields = [part_number]
```

### 3. LangGraph Multi-Part Workflow

다중 파트 요청이면 `agent.py`의 `_run_multi_part_workflow()`가 LangGraph workflow를 실행합니다.

현재 노드:

- `decompose_query`
- `search_each_part`

각 part item은 별도 `RouteDecision`으로 변환되어 검색됩니다. 따라서 첫 번째 파트만 검색하고 나머지를 무시하는 문제가 발생하지 않습니다.

### 4. Single-Part Router

단일 파트 요청이면 `llm_route()`가 다음 항목을 판단합니다.

- intent
- query type: abbreviation, english, korean, mixed, conceptual
- normalized query
- candidate queries
- semantic queries
- requested fields
- vendor/equipment 조건
- 사용할 tool 이름

LLM 호출이 실패하거나 `PART_FINDER_USE_LLM=0`이면 `rule_based_route()`로 fallback합니다.

### 5. Search Tool Layer

`search.py`에는 다음 tool이 있습니다.

- `abbreviation_search_tool`
- `english_name_search_tool`
- `korean_name_search_tool`
- `hybrid_search_tool`
- `semantic_catalog_match_tool`
- `vector_semantic_search_tool`
- `filter_part_rows_tool`
- `aggregate_part_rows_tool`
- `agentic_part_search_tool`

정확한 파트명, 약어, alias, vendor/equipment 조건은 deterministic search로 처리합니다. 의미 기반 표현은 semantic catalog match와 vector search가 보완합니다.

### 6. Canonical Part Family Validation

이번 업데이트에서 추가된 핵심 안정장치입니다.

예:

```bash
python main.py "로봇블레이드 파트넘버. AMAT 장비 기준으로."
```

CSV에 AMAT `Robot Blade`가 없을 때, agent는 AMAT vendor와 fuzzy하게 맞는 다른 row를 반환하지 않습니다.

금지되는 잘못된 결과 예:

- `Gate Valve`
- `Cooling Plate`
- `Shower Head`

검증 순서:

1. 사용자 표현을 canonical part family로 정규화
2. 후보 row의 `part_name`, `description`, base part name이 해당 family와 맞는지 확인
3. vendor/equipment 조건 적용
4. 조건에 맞는 결과가 없으면 no result 또는 다른 조건 후보 안내

### 7. Vector Semantic Search

`vector_index.py`는 `part_numbers.csv`의 row를 RAG chunk로 사용합니다.

row chunk에는 다음 값이 포함됩니다.

- `part_number`
- `part_name`
- `description`
- `equipment_module`
- `vendor_part_number`
- `vendor`
- alias text
- base part name
- semantic hints

현재는 외부 vector DB 없이 TF-IDF vector index를 메모리에 생성합니다. 실제 운영 데이터가 커지면 FAISS, Chroma, pgvector 또는 사내 vector DB로 확장하는 것이 좋습니다.

### 8. Semantic Candidate Behavior

사용자가 정확한 파트명을 말하지 않고 기능/문맥으로 질문하면 agent는 semantic 후보를 찾습니다.

예:

```bash
python main.py "ASML equipment robot arm pick part number"
```

`robot arm pick`, `wafer transfer`, `end effector` 같은 표현은 `Robot Blade` 후보로 이어질 수 있습니다.

현재 final responder는 single-turn CLI 사용성을 위해 불필요한 follow-up 질문을 억제하고, 가능한 후보를 간결하게 보여주는 방향으로 동작합니다.

### 9. Final Responder

`final_responder.py`가 최종 답변을 생성합니다.

LLM이 켜져 있으면 후보 row를 JSON payload로 넘기고 자연스러운 답변을 생성합니다. prompt는 다음을 강제합니다.

- `candidate_rows` 안의 Part Number만 사용
- Part Number 생성, 수정, 보완 금지
- 낮은 confidence나 semantic/vector 후보는 closest candidate로 설명
- 후보 밖 Part Number가 나오면 fallback 답변 사용

## 질문 예시

약어 검색:

```bash
python main.py "W/Q part number 알려줘"
```

영문명 검색:

```bash
python main.py "Liner Quartz part number 알려줘"
```

오타/발음 기반 검색:

```bash
python main.py "Owe ling part number"
```

vendor 조건 검색:

```bash
python main.py "Lam Research 장비 Vacuum Gauge part number 알려줘"
```

equipment/module 조건 검색:

```bash
python main.py "Endura 모델 Slit Valve part number"
```

의미 기반 검색:

```bash
python main.py "ASML equipment robot arm pick part number"
```

다중 파트 검색:

```bash
python main.py "AMAT 기준으로 터보펌프랑 로봇블레이드 파트넘버 알려줘"
python main.py "파트넘버 알려줘 W/Q 그리고 L/Q AMAT"
```

상세 정보 요청:

```bash
python main.py "Vacuum Gauge vendor와 equipment도 알려줘"
```

## 실제 데이터로 교체

데이터 로딩 우선순위는 다음과 같습니다.

1. 명시적으로 전달한 path
2. 프로젝트 루트의 `Part_Number.db`
3. `data/part_numbers.csv`
4. 프로젝트 루트의 `*.csv`
5. 프로젝트 루트의 `*.txt`

권장 CSV schema:

```csv
part_number,part_name,description,equipment_module,vendor_part_number,vendor
P2100958,PM Kit,PM Kit for Helios XP,Helios XP,AMAT-PMK-2100958,AMAT
```

필수 또는 권장 column:

- `part_number`: 필수, `P\d{7}` 형식
- `part_name` 또는 `description`: 둘 중 하나 필수
- `vendor_part_number`: 선택
- `vendor`: 선택
- `equipment_module`: 선택

`data_loader.py`는 다음 column alias를 지원합니다.

- `part_number`: `part_number`, `part no`, `part_no`, `pn`, `p/n`, `partnumber`
- `part_name`: `part_name`, `part name`, `name`, `item_name`, `item name`
- `description`: `description`, `desc`, `part_description`, `part description`
- `vendor_part_number`: `vendor_part_number`, `vendor part number`, `vpn`, `vendor_pn`, `vendor pn`
- `vendor`: `vendor`, `maker`, `manufacturer`, `supplier`
- `equipment_module`: `equipment_module`, `equipment module`, `module`, `tool_module`

## TXT to CSV / RAG Index

사내 원천 데이터가 TXT라면 먼저 CSV로 전처리합니다.

```bash
python scripts/preprocess_txt_to_csv.py company_parts.txt --output data/part_numbers.csv
```

row chunk RAG index 생성:

```bash
python scripts/build_vector_index.py data/part_numbers.csv --output data/part_vector_index.json
```

사내 embedding provider가 있으면 다음 형태로 연결합니다.

```bash
python scripts/build_vector_index.py data/part_numbers.csv --embedding-provider company_embeddings:embed_texts
```

embedding provider는 `list[str]`를 받아 `list[list[float]]`를 반환하는 Python callable이어야 합니다.

## Alias 관리

사용자 표현과 실제 catalog name을 연결하려면 `data/aliases.json`을 수정합니다.

예:

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz"],
  "O-ring": ["o ring", "oring", "owe ling"],
  "Robot Blade": ["로봇블레이드", "로봇 블레이드", "end effector"]
}
```

Alias는 빠르고 정확한 deterministic matching을 위한 보조 데이터입니다. alias에 없는 표현은 LLM router와 vector semantic search가 보완합니다.

## Semantic Hints 관리

기능/문맥 기반 검색 품질을 높이려면 `data/semantic_hints.json`을 수정합니다.

예:

```json
{
  "Robot Blade": ["robot arm", "wafer transfer", "pick", "blade", "end effector"],
  "Throttle Valve": ["pressure control", "flow control", "gas flow"]
}
```

실제 실패 로그(`data/search_failures.jsonl`)를 보고 alias와 semantic hints를 지속적으로 보강하는 것이 좋습니다.

## 사내 Claude Code 참고 문서

`claude_ref.md`는 사내 Claude Code에게 전달할 참고 문서입니다.

포함 내용:

- 현재 프로젝트 구조
- Ollama 기반 LLM을 API-key 기반 `ChatOpenAI` 방식으로 바꿀 때 검토할 파일
- GLM-5.1 또는 gemma4 같은 사내 모델 적용 시 유의사항
- TXT DB를 CSV와 RAG index로 전환하는 절차
- `.env` 변수값
- 유지해야 할 safety rule

민감정보가 포함될 수 있으므로 외부 공유에 주의해야 합니다.

## 테스트

전체 테스트:

```bash
uv run pytest
```

현재 테스트는 다음을 검증합니다.

- dummy data 500 row 생성
- Part Number 형식과 중복 제거
- alias normalization
- fuzzy search deterministic behavior
- vendor/equipment filtering
- semantic catalog matching
- CSV row chunk vector semantic search
- semantic candidate behavior
- multi-part LangGraph workflow
- canonical part family drift 방지
- final responder fallback and hallucination guard
- Langfuse tracing fallback
- config normalization

마지막 확인 결과:

```text
45 passed
```

`.pytest_cache` 쓰기 권한 경고가 표시될 수 있습니다. 테스트 자체가 통과하면 기능 검증에는 영향이 없습니다.

## 운영 전 참고 사항

현재 vector search는 PoC에 맞춘 로컬 TF-IDF 방식입니다. 운영 데이터가 커지거나 의미 검색 품질이 중요해지면 다음 순서로 확장하는 것이 좋습니다.

1. `part_numbers.csv`에 실제 description/spec 컬럼 보강
2. `aliases.json`과 `semantic_hints.json`을 실제 사용자 표현 기반으로 확장
3. 사내 embedding model 또는 API 연결
4. FAISS, Chroma, pgvector, 사내 vector DB 중 하나로 persistent vector index 구성
5. semantic score calibration 재조정
6. 실패 로그 기반 regression test 추가

현재 역할 분리:

- query decomposition/router agent: `llm_router.py`
- LangGraph orchestration: `agent.py`
- tool/search agent: `search.py`, `vector_index.py`
- candidate validation: `agent.py`
- final response agent: `final_responder.py`
