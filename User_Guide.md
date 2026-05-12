# Part Number Finder Agent User Guide

로컬 CSV 또는 SQLite DB에서 Part Name, 약어, 한글 발음, 영문명을 검색해 `P\d{7}` 형식의 Part Number를 반환하는 PoC 사용 가이드입니다.

## 현재 진행사항

이 프로젝트는 RAG나 외부 벡터 DB 없이 **LLM router + deterministic finder tools** 구조로 구현되어 있습니다.

현재 완료된 주요 항목은 다음과 같습니다.

- CLI 진입점: `main.py`
- 패키지 코드: `src/part_finder`
- 더미 데이터 생성 스크립트: `scripts/generate_dummy_data.py`
- 기본 검색 데이터: `data/part_numbers.csv`
- alias 설정: `data/aliases.json`
- 검색 실패 로그: `data/search_failures.jsonl`
- 테스트 코드: `tests`
- 환경 설정 예시: `.env`

구현된 기능은 다음과 같습니다.

1. 사용자 질문에서 Part Name 후보를 추출합니다.
2. 약어, 영문, 한글 발음, 혼합 질의를 분류합니다.
3. LLM router가 켜져 있으면 Ollama 모델로 질의 의도와 검색 도구를 선택합니다.
4. LLM 호출이 실패하거나 비활성화되어 있으면 rule-based router로 자동 fallback합니다.
5. 실제 Part Number 선택은 항상 deterministic search tool이 수행합니다.
6. `P\d{7}` 형식의 Part Number만 결과로 사용합니다.
7. 중복 Part Number는 제거합니다.
8. 낮은 신뢰도 또는 검색 실패 케이스는 `data/search_failures.jsonl`에 기록합니다.
9. 의미 기반 추론이 필요한 질의는 바로 확정하지 않고 사용자 확인을 요청합니다.

## 전체 구조

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── User_Guide.md
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
│       ├── formatter.py
│       ├── llm_router.py
│       ├── normalizer.py
│       ├── search.py
│       └── tracing.py
└── tests
    ├── test_agent.py
    ├── test_config.py
    ├── test_dummy_data.py
    ├── test_formatter.py
    ├── test_normalizer.py
    └── test_search.py
```

## 실행 준비

Python 3.13 이상을 사용합니다.

의존성은 `uv` 기준으로 관리됩니다.

```bash
uv sync
```

일반 Python 환경에서 실행할 경우 `pyproject.toml`의 dependencies를 설치한 뒤 실행하면 됩니다.

## 기본 실행

단일 질문 실행:

```bash
python main.py "W/Q part number 알려줘"
```

반환 개수를 조정하려면 `--top-k`를 사용합니다.

```bash
python main.py --top-k 5 "PM Kit part number 알려줘"
```

대화형 모드:

```bash
python main.py
```

종료하려면 다음 중 하나를 입력합니다.

```text
q
quit
exit
```

## LLM Router 설정

기본 설계는 Ollama 기반 LLM router를 먼저 사용하고, 실패 시 rule-based router로 fallback하는 방식입니다.

`.env`에 다음 값을 설정할 수 있습니다.

```env
PART_FINDER_USE_LLM=1
OLLAMA_MODEL=gemma3:4b
```

LLM router를 끄고 deterministic rule-only 모드로 실행하려면 다음처럼 설정합니다.

```env
PART_FINDER_USE_LLM=0
```

Ollama host는 `OLLAMA_HOST`를 사용합니다. 기존 `.env` 오타 호환을 위해 `OLLMA_HOST`도 읽도록 구현되어 있습니다.

```env
OLLAMA_HOST=http://localhost:11434
```

`http://localhost:11434/v1`처럼 OpenAI compatible path가 붙어 있으면 내부에서 `http://localhost:11434`로 정규화합니다.

## LangSmith Tracing 설정

LangSmith tracing은 선택 기능입니다.

```env
PART_FINDER_TRACE_LANGSMITH=1
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=part-number-finder
```

로컬 테스트나 운영 중 LangSmith 전송을 끄려면 다음처럼 설정합니다.

```env
PART_FINDER_TRACE_LANGSMITH=0
```

검색 실패 로그는 LangSmith 설정과 별개로 `data/search_failures.jsonl`에 저장됩니다.

## 검색 동작

검색 흐름은 다음 순서로 진행됩니다.

1. `agent.py`가 사용자 질문을 받습니다.
2. `llm_router.py`가 질의 유형, 의도, 후보 Part Name, vendor/module 조건, 사용할 tool을 결정합니다.
3. `normalizer.py`가 약어와 alias를 표준 Part Name으로 정규화합니다.
4. `search.py`가 CSV/DB row를 대상으로 fuzzy search, semantic match, filter, aggregate 검색을 수행합니다.
5. `formatter.py`가 한국어 응답 템플릿으로 최종 답변을 만듭니다.
6. score가 낮으면 안전 메시지를 반환하고 실패 로그를 남깁니다.

현재 구현된 search tool은 다음과 같습니다.

- `abbreviation_search_tool`
- `english_name_search_tool`
- `korean_name_search_tool`
- `hybrid_search_tool`
- `semantic_catalog_match_tool`
- `filter_part_rows_tool`
- `aggregate_part_rows_tool`

## 지원 질의 예시

약어 검색:

```bash
python main.py "W/Q part number 알려줘"
```

영문명 검색:

```bash
python main.py "Liner Quartz part number 알려줘"
```

한글 발음 또는 alias 검색:

```bash
python main.py "오링 part number 알려줘"
```

오타/발음형 영문 검색:

```bash
python main.py "Owe ling part number"
```

vendor 조건 검색:

```bash
python main.py "Lam Research 장비 Vacuum Gauge part number 알려줘"
```

module/model 조건 검색:

```bash
python main.py "Endura 모델 Slit Valve part number"
```

의미 기반 검색:

```bash
python main.py "ASML equipment robot arm pick part number"
```

의미 기반 검색은 `Robot Blade`처럼 추론 후보가 나오면 먼저 확인을 요청하고, 대화형 모드에서 `확인`, `yes`, `y`, `ok` 등을 입력하면 최종 Part Number를 반환합니다.

## 더미 데이터 생성

기본 생성:

```bash
python scripts/generate_dummy_data.py
```

이미 `data/part_numbers.csv`가 있으면 덮어쓰지 않고 경고를 출력합니다.

강제 재생성:

```bash
python scripts/generate_dummy_data.py --force
```

생성 결과:

```text
data/part_numbers.csv
```

현재 더미 데이터는 500개 row를 생성하며, Part Number는 `P\d{7}` 형식을 따르고 중복되지 않도록 구성되어 있습니다.

## 실제 데이터로 교체

데이터 로딩 우선순위는 다음과 같습니다.

1. 명시적으로 전달된 path
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
- `vendor_part_number` 또는 `vpn`: 선택
- `vendor`: 선택
- `equipment_module`: 선택

column명이 조금 달라도 `data_loader.py`의 column mapping에서 다음 alias를 지원합니다.

- `part_number`: `part_number`, `part no`, `part_no`, `pn`, `p/n`, `partnumber`
- `part_name`: `part_name`, `part name`, `name`, `item_name`, `item name`
- `description`: `description`, `desc`, `part_description`, `part description`
- `vendor_part_number`: `vendor_part_number`, `vendor part number`, `vpn`, `vendor_pn`, `vendor pn`
- `vendor`: `vendor`, `maker`, `manufacturer`, `supplier`
- `equipment_module`: `equipment_module`, `equipment module`, `module`, `tool_module`

## Alias 관리

사용자 표현과 실제 catalog명을 연결하려면 `data/aliases.json`을 수정합니다.

예시:

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz", "윈도우쿼츠"],
  "O-ring": ["오링", "o ring", "oring", "owe ling"]
}
```

alias는 코드에 내장된 기본 alias와 병합됩니다. 운영 중 실패 로그에 자주 등장하는 표현은 이 파일에 추가하면 검색 품질을 개선할 수 있습니다.

## 테스트

전체 테스트 실행:

```bash
uv run pytest
```

또는 pytest가 설치된 환경에서:

```bash
pytest
```

현재 테스트가 검증하는 범위는 다음과 같습니다.

- 더미 데이터 500 row 생성
- Part Number 형식과 중복 제거
- alias normalization
- 약어/영문/한글/혼합 질의 분류
- fuzzy search 결과의 deterministic behavior
- vendor/module 조건 필터링
- semantic catalog match
- 낮은 confidence 응답 처리
- LangSmith/Ollama 설정 정규화

## 운영 시 참고사항

현재 PoC는 deterministic finder tool이 최종 Part Number를 선택하도록 설계되어 있습니다. LLM은 Part Number를 직접 생성하지 않고, 질의 이해와 route 선택에만 사용됩니다.

검색 confidence 기준은 `agent.py`와 `formatter.py`에서 70점으로 관리됩니다. 이보다 낮은 결과는 Part Number를 확정하지 않고 재입력 안내 메시지를 반환합니다.

실제 운영 데이터로 전환할 때는 다음 순서로 점검하는 것이 좋습니다.

1. CSV 또는 DB column명이 mapping 가능한지 확인합니다.
2. `part_number`가 `P\d{7}` 형식을 따르는지 확인합니다.
3. 대표 query를 `tests`에 추가합니다.
4. 실패 로그를 확인해 alias를 보강합니다.
5. vendor/module 표현이 자주 쓰이면 `llm_router.py`의 alias map을 확장합니다.

## 알려진 주의점

- 현재 README와 일부 소스 문자열은 콘솔에서 한글이 깨져 보일 수 있습니다.
- 새로 작성하는 문서는 UTF-8 기준의 정상 한국어로 작성했습니다.
- TXT 로딩은 CSV처럼 delimiter가 있는 텍스트만 지원합니다.
- 완전한 비정형 문서 검색은 이후 RAG workflow로 분리하는 것이 적합합니다.
