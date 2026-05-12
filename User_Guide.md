# Part Number Finder Agent User Guide

이 문서는 현재 구현된 Part Number Finder Agent PoC의 사용 방법과 내부 동작 방식을 정리합니다.

## 현재 상태

현재 프로젝트는 단순 정규식/alias 기반 finder가 아니라, LLM과 Python tool을 함께 사용하는 agentic finder 구조입니다.

중요한 설계 원칙은 다음과 같습니다.

- LLM은 사용자 질문의 의도와 검색 전략을 판단합니다.
- 실제 Part Number 후보는 Python search tool, CSV/DB, vector index 결과에서만 가져옵니다.
- LLM이 없는 Part Number를 새로 만들어내지 않도록 final confirm 단계를 둡니다.
- 최종 답변은 고정 템플릿이 아니라 chatbot처럼 자연스러운 문장으로 생성합니다.
- LLM이 실패하거나 비활성화되어도 deterministic fallback으로 동작합니다.

## 전체 구성

```text
.
├── main.py
├── README.md
├── User_Guide.md
├── pyproject.toml
├── data
│   ├── aliases.json
│   ├── part_numbers.csv
│   └── search_failures.jsonl
├── scripts
│   └── generate_dummy_data.py
├── src
│   └── part_finder
│       ├── agent.py
│       ├── config.py
│       ├── data_loader.py
│       ├── final_responder.py
│       ├── formatter.py
│       ├── llm_router.py
│       ├── normalizer.py
│       ├── search.py
│       ├── tracing.py
│       └── vector_index.py
└── tests
    ├── test_agent.py
    ├── test_config.py
    ├── test_dummy_data.py
    ├── test_formatter.py
    ├── test_normalizer.py
    └── test_search.py
```

## 실행 준비

Python 3.13 이상을 기준으로 합니다.

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

`OLLAMA_HOST`는 `http://localhost:11434/v1`처럼 OpenAI compatible path가 붙어 있어도 내부에서 `http://localhost:11434`로 정규화합니다. 기존 오타 호환을 위해 `OLLMA_HOST`도 읽습니다.

## LangSmith Tracing

LangSmith tracing은 선택 기능입니다.

```env
PART_FINDER_TRACE_LANGSMITH=1
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=part-number-finder
```

로컬 테스트에서 tracing 전송을 끄려면:

```env
PART_FINDER_TRACE_LANGSMITH=0
```

검색 실패 로그는 LangSmith 설정과 별개로 `data/search_failures.jsonl`에 저장됩니다.

## 내부 동작

### 1. 사용자 입력 수신

`main.py`가 사용자 질문을 받고 `agent.py`의 `answer_query_result`를 호출합니다.

### 2. LLM Router

`llm_router.py`가 다음 항목을 판단합니다.

- 사용자의 intent
- query type: abbreviation, english, korean, mixed
- normalized query
- candidate queries
- semantic candidates
- requested fields
- vendor/equipment 조건
- 사용할 tool 이름

LLM 호출이 실패하거나 `PART_FINDER_USE_LLM=0`이면 rule-based router로 fallback합니다.

### 3. Search Tool Layer

`search.py`에는 다음 tool이 있습니다.

- `abbreviation_search_tool`
- `english_name_search_tool`
- `korean_name_search_tool`
- `hybrid_search_tool`
- `semantic_catalog_match_tool`
- `vector_semantic_search_tool`
- `filter_part_rows_tool`
- `aggregate_part_rows_tool`

정확한 파트명, 약어, alias, 벤더/장비 조건은 deterministic search로 처리합니다.

### 4. Vector Semantic Search

`vector_index.py`는 `part_numbers.csv`의 row를 RAG chunk로 사용합니다.

한 row chunk에는 다음 값이 포함됩니다.

- `part_number`
- `part_name`
- `description`
- `equipment_module`
- `vendor_part_number`
- `vendor`
- base part name
- 제한된 semantic hints

현재는 외부 vector DB 없이 TF-IDF vector index를 메모리에 생성합니다. 따라서 설치와 운영은 단순하지만, 실제 운영 데이터에서는 FAISS, Chroma, pgvector 같은 저장형 vector index로 확장할 수 있습니다.

### 5. Semantic Confirmation

사용자가 정확한 파트명을 말하지 않고 기능/문맥으로 질문하면 agent는 바로 Part Number를 확정하지 않습니다.

예:

```bash
python main.py "ASML equipment robot arm pick part number"
```

응답 예:

```text
말씀하신 표현은 데이터의 정확한 파트명과 완전히 일치하지는 않습니다.
의미상으로는 Robot Blade for Implant Module (P2200013) 쪽이 가장 가까워 보입니다.
이 파트를 찾으신 게 맞으면 '확인'이라고 답해주세요.
```

대화형 모드에서 사용자가 다음처럼 답하면:

```text
확인
```

확인된 후보의 Part Number를 반환합니다.

### 6. Last Confirm

`agent.py`의 `_last_confirm` 단계가 최종 후보를 다시 확인합니다.

- `part_number`가 비어 있지 않은지 확인
- 중복 Part Number 제거
- score가 confidence threshold 이상인지 확인
- 후보 밖의 값을 final responder에 넘기지 않음

### 7. Final Responder

`final_responder.py`는 최종 답변을 생성합니다.

LLM이 켜져 있으면 후보 row를 JSON payload로 넘기고 자연스러운 한국어 답변을 생성합니다. 단, prompt와 후처리에서 다음을 강제합니다.

- `candidate_rows` 안의 Part Number만 사용
- Part Number를 생성, 수정, 보완하지 않음
- confirm 상태가 불확실하면 짧은 확인 질문 생성
- LLM 응답에 후보 Part Number가 없으면 fallback 답변 사용

LLM이 꺼져 있거나 실패하면 안전한 deterministic fallback 답변을 반환합니다.

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

벤더 조건 검색:

```bash
python main.py "Lam Research 장비 Vacuum Gauge part number 알려줘"
```

장비/모듈 조건 검색:

```bash
python main.py "Endura 모델 Slit Valve part number"
```

의미 기반 검색:

```bash
python main.py "ASML equipment robot arm pick part number"
```

상세 정보 요청:

```bash
python main.py "Vacuum Gauge vendor와 equipment도 알려줘"
```

## Dummy Data 생성

기본 생성:

```bash
python scripts/generate_dummy_data.py
```

강제 재생성:

```bash
python scripts/generate_dummy_data.py --force
```

생성 결과:

```text
data/part_numbers.csv
```

현재 dummy data는 500개 row를 생성하며, Part Number는 `P\d{7}` 형식을 따릅니다.

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

## Alias 관리

사용자 표현과 실제 catalog명을 연결하려면 `data/aliases.json`을 수정합니다.

예:

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz"],
  "O-ring": ["o ring", "oring", "owe ling"]
}
```

Alias는 빠르고 정확한 deterministic matching을 위한 보조 데이터입니다. alias에 없는 표현은 LLM router와 vector semantic search가 보완합니다.

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
- semantic confirmation flow
- final responder fallback
- config normalization

마지막 확인 결과:

```text
28 passed
```

`.pytest_cache` 쓰기 권한 경고가 표시될 수 있습니다. 테스트 자체가 통과했다면 기능 검증에는 영향이 없습니다.

## 운영 시 참고 사항

현재 vector search는 PoC에 맞춘 로컬 TF-IDF 방식입니다. 운영 데이터가 커지거나 의미 검색 품질이 더 중요해지면 다음 순서로 확장하는 것이 좋습니다.

1. `part_numbers.csv`의 실제 description/spec 컬럼 보강
2. row chunk 외에 spec 단위 chunk 추가
3. embedding model 도입
4. FAISS, Chroma, pgvector 중 하나로 persistent vector index 구성
5. semantic score calibration 재조정
6. 실제 실패 로그를 기반으로 alias와 semantic hints 보강

멀티에이전트 구조는 현재 논리적으로 분리되어 있습니다.

- router agent 역할: `llm_router.py`
- tool agent 역할: `search.py`, `vector_index.py`
- last confirm agent 역할: `agent.py`의 `_last_confirm`
- final response agent 역할: `final_responder.py`

실제 LangGraph 또는 독립 agent orchestration으로 분리할 수도 있지만, 현재 PoC에서는 hallucination 위험을 줄이기 위해 하나의 제어 흐름 안에서 역할을 나누는 방식을 사용합니다.
