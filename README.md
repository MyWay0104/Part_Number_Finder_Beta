# Part Number Finder Agent PoC

로컬 CSV 또는 SQLite DB에서 Part Name, 약어, 한글명, 영어명을 검색해 `P\d{7}` 형식의 Part Number를 반환하는 deterministic PoC입니다.

## 설계

PoC 단계에서는 RAG/Multi-Agent 대신 **LLM router + deterministic finder tools** 구조를 사용합니다.

1. LLM 또는 rule 기반 router가 사용자 질문에서 part name 후보를 추출합니다.
2. 질문이 약어, 영어 원문, 한글 발음, mixed query인지 판단합니다.
3. `abbreviation_search_tool`, `english_name_search_tool`, `korean_name_search_tool`, `hybrid_search_tool` 중 하나로 분기합니다.
4. finder tool은 CSV/DB의 실제 row를 fuzzy search합니다.
5. `P\d{7}` part_number만 반환하고 중복 part_number를 제거합니다.
6. 한국어 formatter가 최종 답변을 생성합니다.

LLM router는 기본 실행 경로에서 우선 사용됩니다. Ollama 호출이 실패하면 rule-based router로 자동 fallback합니다. LLM은 part number를 직접 생성하지 않고, part name 후보와 tool route만 판단합니다.

```bash
# PowerShell
$env:OLLAMA_MODEL="gemma3:4b"
python main.py "W/Q의 파트넘버 알려줘"
```

또는 `.env`에 추가할 수 있습니다.

```env
PART_FINDER_USE_LLM=1
OLLAMA_MODEL=gemma3:4b
```

LLM router를 끄고 deterministic rule-only fallback만 쓰려면:

```env
PART_FINDER_USE_LLM=0
```

운영 tracing은 `.env`의 LangSmith 설정을 사용합니다.

```env
PART_FINDER_TRACE_LANGSMITH=1
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=part-number-finder
```

폐쇄망 또는 로컬 테스트에서 LangSmith 전송을 끄려면 `PART_FINDER_TRACE_LANGSMITH=0`으로 둡니다. 검색 실패 로그는 LangSmith 전송 여부와 관계없이 로컬 `data/search_failures.jsonl`에 남습니다.

검색 결과가 없거나 best score가 낮으면 `data/search_failures.jsonl`에도 실패 이벤트를 남깁니다. 이 파일은 운영 중 한글 발음 alias, 약어, vendor 표현을 보강하는 입력 데이터로 사용할 수 있습니다.

지원 모델 예:

- `gemma3:4b`

LLM은 query type 분류, normalized query, tool 선택에만 사용합니다. 실제 Part Number 선택은 같은 deterministic finder tool이 수행합니다.

예를 들어 `오링 파트넘버 알려줘`, `Owe ling part number`는 `O-ring` 후보로 정규화된 뒤 CSV row에서 검색됩니다.

## Dummy Data 생성

기본 생성:

```bash
python scripts/generate_dummy_data.py
```

이미 `data/part_numbers.csv`가 있으면 덮어쓰지 않고 경고를 출력합니다.

강제 재생성:

```bash
python scripts/generate_dummy_data.py --force
```

생성 파일:

```text
data/part_numbers.csv
```

## CLI 테스트

```bash
python main.py "윈도우쿼츠 파트넘버 알려줘"
python main.py "피엠킷 혹은 피엠키트의 파트넘버 알려줘"
python main.py "W/Q의 파트넘버 알려줘"
python main.py "Liner Quartz 의 파트넘버 알려줘"
```

Interactive mode:

```bash
python main.py
```

종료는 `q`, `quit`, `exit` 중 하나를 입력합니다.

## Pytest 실행

```bash
pytest
```

`tests/test_dummy_data.py`는 테스트 시작 시 `scripts/generate_dummy_data.py --force`로 500 row dummy data를 재생성합니다.

## 실제 데이터로 교체

우선순위는 다음과 같습니다.

1. `Part_Number.db`
2. `data/part_numbers.csv`
3. 프로젝트 루트의 `.csv`
4. 프로젝트 루트의 `.txt`

권장 CSV schema:

```csv
part_number,part_name,description,equipment_module,vendor_part_number,vendor
P2100958,피엠킷,PM Kit for Helios XP,Helios XP,AMAT-PMK-2100958,AMAT
```

필수 또는 권장 컬럼:

- `part_number`: 필수, `P\d{7}` 형식
- `part_name` 또는 `description`: 둘 중 하나 필수
- `vendor_part_number` 또는 `vpn`: 선택
- `vendor`: 선택
- `equipment_module`: 선택

컬럼명이 조금 달라도 `src/part_finder/data_loader.py`의 column mapping에서 안전하게 매핑합니다.

## Alias 설정

Alias는 코드 내부가 아니라 `data/aliases.json`에서 관리합니다.

예:

```json
{
  "Window Quartz": ["W/Q", "WQ", "window quartz", "윈도우쿼츠"]
}
```
