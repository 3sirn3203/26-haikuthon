# Stage2: OpenAI Batch 프롬프트 생성

Stage1 component를 OpenAI Batch API로 제출해 이미지 생성용 영어 프롬프트를 만든다.
`test_openai`에서 검증한 프롬프트 조립과 Batch 제출·수집 구현을 본 모듈로 옮겼다.
기본 모델은 **`gpt-6.1-terra`**, reasoning은 **`none`**, temperature는 **`1.4`**다.
`max_output_tokens=4096`, `store=False`, Batch 처리 기한은 `24h`다.
추론 토큰과 최종 텍스트는 같은 출력 토큰 한도를 사용한다.

CPU 환경에서 실행하며, import와 `--dry-run`은 API 호출이나 결과 저장을 수행하지 않는다.
`test_openai/`와 기존 실험 결과는 비교용으로 남겨 두었다. 앞으로 본 구현의 프롬프트는
**`stage2_prompt_generation/prompts.py`**에서 수정한다.

## 설치와 기본 설정

프로젝트 루트에서 실행한다.

```bash
.venv/bin/python -m pip install -r stage2_prompt_generation/requirements.txt
```

루트 `.env` 또는 환경변수에 `OPENAI_API_KEY`를 설정한다. 기존 환경변수가 우선한다.
`config.yaml`은 입력·출력 디렉터리, `component_id`, 모델, reasoning, temperature, 출력 토큰 한도,
문체 배정 seed를 지정한다. `--config PATH`로 다른 설정 파일을 사용할 수 있다.
상대 입력·출력 경로는 프로젝트 루트를 기준으로 해석한다.

기본 입력은 `outputs_stage1/train_components_v5.jsonl`과
`outputs_stage1/test_components_v5.jsonl`이다. `component_id`를 바꾸면 Stage1 파일명과
기본 출력 파일명에 같은 접미사가 적용된다. `null`이면 접미사 없는 파일명을 사용한다.

## 제출과 수집

train/test는 각각 별도의 Batch로 제출한다. 아래 명령은 제출 후 Batch ID를 출력하고 종료한다.

```bash
.venv/bin/python -m stage2_prompt_generation --split train
.venv/bin/python -m stage2_prompt_generation --split test
```

기본값은 `--split test`다. 각 제출에서 출력한 ID로 상태 확인 및 수집을 수행한다.
Batch에 저장된 split을 복원하므로 수집할 때 `--split`을 생략할 수 있다.

```bash
.venv/bin/python -m stage2_prompt_generation --batch-id batch_...
# 완료될 때까지 대기한 뒤 수집
.venv/bin/python -m stage2_prompt_generation --batch-id batch_... --wait
```

미완료면 상태만 출력하고 종료한다. `--wait`는 기본 30초 간격으로 확인하며
`--poll-interval`로 1~60초 사이에서 바꿀 수 있다. 제출 명령에도 `--wait`를 붙일 수 있다.
실행 중 대기를 중단해도 이미 제출한 Batch는 계속 진행된다.

작은 범위를 검사하거나 다른 입력을 사용하려면 다음과 같이 지정한다.

```bash
# 실제 API 호출 없이, cluster 0의 요청 JSONL을 표준 출력으로 확인
.venv/bin/python -m stage2_prompt_generation --cluster-id 0 --dry-run

# cluster 0 전체를 별도 결과 파일로 제출
.venv/bin/python -m stage2_prompt_generation --cluster-id 0 \
  --output outputs_stage2/test_cluster_0_terra_low.jsonl

# 위 제출 결과를 수집
.venv/bin/python -m stage2_prompt_generation --batch-id batch_... \
  --output outputs_stage2/test_cluster_0_terra_low.jsonl
```

`--id`는 여러 번 지정할 수 있다. `--limit`, `--group-id`, `--cluster-id`와는 상호 배타적이다.
`--offset N --limit M`은 입력의 처음 N개 행을 건너뛰고 다음 M개를 선택한다.
`--offset`은 ID 기준이 아닌 행 기준이며, `--id`/`--group-id`/`--cluster-id`와 함께 사용할 수 없다.
수집할 때 선택 옵션은 다시 지정하지 않는다. `--input PATH`로 별도 Stage1 입력을 사용했다면
수집할 때도 같은 `--input`을, `--output PATH`를 지정했다면 같은 `--output`을 사용한다.
입력 데이터가 제출 후 바뀌면 component 해시 검사에서 거부된다.

여러 Batch의 결과를 한 파일에 누적하려면 제출과 수집에 모두 `--append`를 지정한다.
기존 파일을 검증하고, 이미 저장된 ID의 재제출은 API 호출 전에 거부한다.
수집에서는 내용과 메타데이터까지 같은 ID를 건너뛰므로 같은 Batch를 재수집해도 중복되지 않는다.
같은 ID의 내용이 다르거나 기존 파일이 잘못됐으면 기존 파일과 원격 입력을 보존하고 중단한다.
병합은 임시 파일에 완성한 뒤 원자적으로 교체한다. 동시에 같은 파일에 수집하면 한 실행은
잠금 오류로 중단하므로, 해당 실행을 나중에 다시 수집한다. 정상 종료 시 잠금 파일은 제거된다.

```bash
# v5 입력은 ID 순서이므로 이 범위는 ID 4000~7999에 해당한다.
.venv/bin/python -m stage2_prompt_generation --split test --offset 4000 --limit 4000 \
  --input outputs_stage1/test_components_v5.jsonl \
  --output outputs_stage2/test_prompts_v3.jsonl --append

# 해당 Batch를 수집할 때는 offset/limit 없이 같은 누적 파일을 지정한다.
.venv/bin/python -m stage2_prompt_generation --batch-id batch_... \
  --input outputs_stage1/test_components_v5.jsonl \
  --output outputs_stage2/test_prompts_v3.jsonl --append
```

`--model`, `--reasoning-effort`, `--temperature`, `--max-output-tokens`, `--style-seed`로 설정을 덮어쓸 수 있다.
Temperature는 0~2 범위이며 reasoning이 `none`일 때만 지정할 수 있다.
Reasoning을 켜려면 `--reasoning-effort low --temperature none`처럼 temperature를 함께 생략한다.
설정 파일에서도 `temperature: null`로 생략할 수 있다. 수집할 때 생성 옵션은 다시 지정하지 않는다.

## 저장 형식

기본 결과 파일은 다음 두 개다. 기본 동작은 **기존 파일을 덮어쓰지 않는다.**
별도 실험은 새 `--output`을 지정하고, 기존 결과에 이어 수집하려면 `--append`를 사용한다.

```text
outputs_stage2/train_prompts_v5.jsonl
outputs_stage2/test_prompts_v5.jsonl
```

Train은 정확히 세 필드를 저장한다.

```json
{"id": "0000", "prompt": "An image-generation prompt.", "group_id": 0}
```

Test는 정확히 다섯 필드를 저장한다.

```json
{"id": "0000", "prompt": "An image-generation prompt.", "group_id": 667, "cluster_id": 0, "is_query": true}
```

- `id`는 Stage1의 `prompt_id` 문자열을 그대로 보존한다.
- `group_id`, `cluster_id`, `is_query`는 원본 component의 값이다. ID들은 0 이상의 정수,
  `is_query`는 boolean이어야 하며, 누락·잘못된 타입은 제출 전에 거부한다.
- `is_query=true`는 평가 질의로 선택된 test 항목이다. `false`도 이미지 후보이므로 모두 생성·저장한다.
  v5 전체는 train 4,000개, test 16,000개이며 test는 query 8,000개와 나머지 후보 8,000개를 포함한다.
- Train 입력의 `cluster_id: null`은 출력하지 않는다. 원본 component, API 응답, usage,
  reasoning, 문체 ID, 검수·요약 파일, 별도 `test_queries.jsonl`은 저장하지 않는다.
- Stage3는 위 형식과 기존 `id`/`prompt` 두 필드 형식을 모두 읽는다. 기본 입력도 v5로 연결했다.
  평가 질의는 test 결과에서 `is_query=true`로 선택한다.

## 프롬프트와 문체 배정

`prompts.py`의 `SYSTEM_PROMPT`, `STYLE_AXES`, `build_prompt()`에 조립 로직을 모았다.
최근 Terra 실험과 지시문·입력 조립을 동일하게 유지한다.

- 모든 객체·수량·속성 결합·관계·배경·매체·시점·조명을 보존하고, 주어의 참조를 명확히 한다.
- 제공되지 않은 시각적 사실이나 설명을 추가하지 않고 180단어 이내의 영어 plain text만 반환한다.
- 관계는 `"ball inside suitcase"`와 같은 문자열 배열이다. category가 중복될 때만 구분 가능한
  최소 속성을 붙이며, 구분할 수 없으면 제출 전에 거부한다.
- 문체는 시작 정보 4종 × 관계 구문 4종 × 렌더링 정보 배치 2종 × 속성 위치 2종의 64가지다.
  group/cluster 안에서 균형 있게 배정하며, component 내용이나 `is_query` 값으로 문체를 고르지 않는다.
- 일부 ID·group·cluster만 선택해도 **전체 입력을 기준으로** 배정하므로 전체 실행과 지시문이 같다.
  관계가 하나뿐이면 해당 구문 지시를 조정하고, 속성이 없으면 속성 위치 지시를 생략한다.

## Batch 저장과 오류 처리

요청 JSONL은 메모리에서 조립해 업로드하고 제출 성공 직후 참조를 해제한다. 로컬 요청·job 파일은 없다.
원격 입력 파일은 진행 중인 Batch가 사용하므로 유지하다가, 종료 후 결과 저장에 성공하면 삭제한다.

서버의 결과 순서와 관계없이 `custom_id`로 결합하고 원본 입력 순서로 저장한다.
HTTP 오류, refusal, incomplete, 빈 응답, 누락된 결과는 실패 ID로 출력한다.
부분 실패 시 성공한 결과만 저장하고 종료 코드 1을 반환한다. 자동 재시도나 재제출은 하지 않는다.
결과는 임시 파일에 완성한 뒤 기본 모드에서는 덮어쓰기 없이 게시하고,
`--append`에서는 기존 행을 포함한 병합 파일로 원자적으로 교체한다.
저장 실패 시 원격 입력을 남겨 재수집할 수 있다.

## 검증

외부 API를 호출하지 않는 HTTP mock과 문체·출력·Stage3 호환성 테스트:

```bash
.venv/bin/python -m unittest stage2_prompt_generation.test_runner \
  stage2_prompt_generation.test_styles stage3_image_generation.test_inputs
```
