# Part Number Finder Agent PoC

`Part Number Finder`는 CSV 또는 SQLite DB에 저장된 파트 카탈로그에서 사용자의 자연어 질문을 해석하고, 실제 데이터에 존재하는 `P\d{7}` 형식의 Part Number를 찾아주는 Agentic PoC입니다.

기존의 정규식, alias, 고정 템플릿 중심 검색에서 다음 구조로 확장되었습니다.

- LLM router가 사용자 의도와 검색 전략을 먼저 판단
- 정확한 키워드/파트명은 deterministic Python search tool로 검색
- 정확 매칭이 약하면 CSV row chunk 기반 vector semantic search 사용
- LLM은 파트넘버를 직접 생성하지 않고, tool이 반환한 후보만 해석
- 마지막 confirm 단계에서 후보가 실제 tool 결과인지 재검증
- 최종 답변은 고정 문장이 아니라 LLM final responder가 자연스러운 chatbot 형태로 생성

## Architecture

```text
User Input
  ↓
LLM Router / Rule Router
  - intent 판단
  - query type 판단
  - normalized query 생성
  - 사용할 search tool 선택

  ↓
Search Tool Layer
  - exact / fuzzy keyword search
  - abbreviation / Korean pronunciation search
  - vendor / equipment filter
  - CSV row chunk vector semantic search

  ↓
Candidate Reasoning
  - 후보 score 비교
  - semantic 후보는 사용자 확인 요청

  ↓
Last Confirm
  - part_number 존재 여부 확인
  - confidence threshold 확인
  - 후보 밖의 part_number 차단

  ↓
Final Responder
  - tool 결과만 사용
  - 자연스러운 chatbot 답변 생성
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
data/search_failures.jsonl
```

## Search Flow

1. `agent.py`가 사용자 질문을 받습니다.
2. `llm_router.py`가 LLM 또는 rule 기반으로 intent, query type, 후보 query, tool을 결정합니다.
3. 정확한 파트명, 약어, alias, 벤더/장비 조건은 `search.py`의 deterministic tool이 처리합니다.
4. 정확 매칭이 부족하거나 의미 기반 요청이면 `vector_index.py`가 `part_numbers.csv` row를 chunk로 보고 TF-IDF vector search를 수행합니다.
5. semantic 후보는 바로 확정하지 않고, 먼저 사용자에게 “이 파트가 맞는지” 확인을 요청합니다.
6. 사용자가 `확인`, `yes`, `y`, `ok` 등으로 답하면 실제 Part Number를 반환합니다.
7. 최종 응답은 `final_responder.py`가 생성합니다. LLM이 활성화되어 있으면 자연어로 감싸고, 비활성화되거나 실패하면 안전한 fallback 문장을 사용합니다.

## LLM Safety Policy

LLM은 Part Number를 직접 만들 수 없습니다.

- Part Number 후보는 CSV/DB/vector search 결과에서만 옵니다.
- final responder는 `candidate_rows`에 포함된 Part Number만 사용할 수 있습니다.
- LLM 답변에 후보 Part Number가 포함되지 않으면 fallback 답변으로 대체합니다.
- `_last_confirm` 단계에서 score와 Part Number 존재 여부를 확인합니다.

## Vector/RAG

현재 구현은 외부 vector DB 없이 로컬 TF-IDF vector index를 사용합니다.

- chunk 단위: `part_numbers.csv`의 row 1개
- chunk 내용: `part_number`, `part_name`, `description`, `equipment_module`, `vendor_part_number`, `vendor`, semantic hints
- index 파일: 런타임 메모리 캐시
- 검색 함수: `vector_semantic_search_tool`

예를 들어 사용자가 다음처럼 말하면:

```bash
python main.py "ASML equipment robot arm pick part number"
```

정확한 파트명이 없어도 의미적으로 `Robot Blade` 후보를 찾고, 먼저 확인 질문을 반환합니다.

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

LangSmith tracing은 선택 사항입니다.

```env
PART_FINDER_TRACE_LANGSMITH=1
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=part-number-finder
```

## Run

의존성 설치:

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

## Dummy Data

기본 dummy data 생성:

```bash
python scripts/generate_dummy_data.py
```

강제 재생성:

```bash
python scripts/generate_dummy_data.py --force
```

생성 파일:

```text
data/part_numbers.csv
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

1. 명시적으로 전달된 path
2. 프로젝트 루트의 `Part_Number.db`
3. `data/part_numbers.csv`
4. 프로젝트 루트의 `*.csv`
5. 프로젝트 루트의 `*.txt`

## Alias

사용자 표현과 실제 카탈로그명을 연결하려면 `data/aliases.json`을 수정합니다.

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz"],
  "O-ring": ["o ring", "oring", "owe ling"]
}
```

Alias는 deterministic exact/fuzzy search의 보조 수단입니다. alias에 없는 표현은 LLM router와 vector semantic search가 보완합니다.

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
- final responder fallback
- agent confirmation flow
- config normalization

마지막 확인 결과:

```text
28 passed
```

`.pytest_cache` 쓰기 권한 경고가 표시될 수 있지만, 테스트 결과에는 영향이 없습니다.
