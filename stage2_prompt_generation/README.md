# Stage2 프롬프트 생성과 검수

Stage1의 모든 프롬프트를 먼저 생성·저장한 다음, 모든 프롬프트를 기존 source critic으로
한 번씩 검수한다. 생성과 검수 사이에 재생성, 추가 검수, HTTP 재시도는 없다.
train 4,000개 + test 8,000개의 전체 실행이 완료되면 LLM 요청은 정확히 24,000회다.

패키지를 import하는 것만으로 다운로드, 서버 실행, 출력 저장을 수행하지 않는다.

## 환경 준비

Linux/NVIDIA CUDA GPU와 Python 3.12 전용 환경을 사용한다.

```bash
python3.12 -m venv .venv-stage2
.venv-stage2/bin/python -m pip install -r stage2_prompt_generation/requirements.txt
```

모델은 `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16이다.
의존성은 자동 설치하지 않는다. 서버 실행 시 vLLM이 체크포인트를 캐시에서 읽거나 다운로드한다.
현재 프로젝트에 있는 캐시를 사용하려면 다음 환경변수를 지정한다.

```bash
export HF_HOME=/workspace/haikuthon/.cache/huggingface
export HF_HUB_CACHE=/workspace/haikuthon/.cache/huggingface/hub
```

## 실행 순서

1. `outputs_stage1/train_components.jsonl`, `test_components.jsonl`의 입력 전체를 검사한다.
2. train 전체, test 전체를 각각 **한 번씩 생성**해 프롬프트 JSONL에 저장한다.
3. 두 split의 생성 파일을 모두 닫은 뒤, 저장된 프롬프트를 읽어 train 전체, test 전체를 각각
   **한 번씩 검수**한다. 기존 critic에 원본 scene과 생성 프롬프트를 함께 전달한다.
4. 검수 결과와 split별 집계를 저장하고 종료한다. 검수 실패 항목도 프롬프트 파일에 남긴다.

생성 JSON의 `prompt_text` 문자열만 꺼내 모든 행을 저장한다. 응답이 비어 있거나 JSON 형식이 깨진 경우에도
행을 누락하지 않고 원문을 `prompt`에 보존해 이후 critic에 전달한다.

검수는 기존 critic의 `{"issues": [...]}` 응답만 사용한다. 정확히 `issues` 키 하나와 빈 배열을
반환하면 통과다. issue가 있거나 응답 형식이 잘못되면 실패로 기록한다. 재실행하지 않는다.

생성은 `sentence_swap` 지시문으로 조건을 자연스러운 영어 문장으로 서술한다.
기본 생성은 thinking 활성화, reasoning effort `low`, temperature 1.0, top_p 0.95,
top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0이다.
생성 최대 출력 2,048토큰에는 추론과 최종 응답이 모두 포함된다.
검수는 thinking 비활성화, temperature 0.0, 최대 출력 384토큰으로 유지한다.
`generation.enable_thinking`과 `critic.enable_thinking`이 각각 모델의 기본값보다 우선한다.

JSON 스키마를 공백 허용 xgrammar EBNF로 변환해 요청에 전달한다. JSON 필드와 타입은
강제하면서 기존 서버의 `disable_any_whitespace=true` 설정에도 공백 토큰을 허용한다.
서버를 재시작할 필요가 없으며, 새로 시작하는 서버의 기본값도 공백 허용이다.

문체 가이드는 정보 순서 5종 × 문장 구성 4종의 20종이다. 정보 순서는 배경/물체/관계/매체/시점,
문장 구성은 직접 서술/관계절/병렬절/장단문 혼합이다. Stage1의 family를 복원해 8개 항목에
서로 다른 가이드를 배정한다. 각 family에서 정보 순서는 1~2회씩, 문장 구성은 2회씩 사용한다.
seed와 split 및 family의 ID로 배정을 재현하며 입력 순서와 scene 내용은 바꾸지 않는다.
독립 장면은 seed·split·ID의 해시로 배정한다. 가이드 다양화가 실제 출력 다양성을 보장하는
것은 아니므로 생성 결과의 문장 구조는 별도로 확인해야 한다.

생성과 검수의 기본 동시 요청은 각각 16개다. 서버는 context 8,192, 최대 sequence 32,
GPU 메모리 목표 85%를 사용한다.

## 출력

`outputs_stage2`에 다음 6개 파일을 저장한다. 입력 순서와 네 자리 ID를 유지하며,
train/test의 같은 ID는 서로 다른 항목으로 취급한다. JSONL은 한 행씩 즉시 flush한다.

```text
train_prompts.jsonl
test_prompts.jsonl
train_verification.jsonl
test_verification.jsonl
train_summary.json
test_summary.json
```

프롬프트 행은 기존과 동일하게 `id`, `prompt`만 포함한다.

```json
{"id":"0000","prompt":"A photograph shows two red mugs beside one blue bowl ..."}
```

검수 행은 정확히 `id`, `ok`, `reply`만 포함한다. `ok`는 JSON boolean이고,
`reply`는 검수 LLM의 응답 원문 문자열이다. 파싱에 실패한 응답도 그대로 보존한다.

```json
{"id":"0000","ok":true,"reply":"{\"issues\":[]}"}
{"id":"0001","ok":false,"reply":"{\"issues\":[\"Incorrect object count\"]}"}
```

요약 파일은 해당 split의 검수가 완료되면 저장하며, `pass_rate`는 0~1 사이 비율이다.

```json
{"total":4000,"passed":3500,"failed":500,"pass_rate":0.875}
```

새 실행은 선택한 split의 기존 프롬프트·검수·요약 파일을 교체한다. 미완료 실행에 이전 요약이
남지 않도록 시작 시 해당 파일들을 제거한다. `--limit`도 선택한 split의 결과 전체를 교체한다.

## 명령

프로젝트 루트에서 실행한다. 이미 실행 중인 vLLM 서버에는 `--external-server`를 사용한다.
이 옵션은 외부 서버를 시작하거나 종료하지 않는다.

```bash
# 기존 서버로 전체 생성 → 전체 검수 (12,000 + 12,000 LLM 요청)
.venv-stage2/bin/python -m stage2_prompt_generation generate --external-server

# 각 split 2개씩: 4개 생성 완료 후 4개 검수 (총 8 LLM 요청)
.venv-stage2/bin/python -m stage2_prompt_generation generate --external-server --limit 2

# train만 전체 생성한 뒤 검수 (4,000 + 4,000 LLM 요청)
.venv-stage2/bin/python -m stage2_prompt_generation generate --external-server --split train

# 서버 시작 → 전체 생성 → 전체 검수 → 직접 시작한 서버 종료
.venv-stage2/bin/python -m stage2_prompt_generation generate

# 서버만 실행하거나 실행 명령만 확인
.venv-stage2/bin/python -m stage2_prompt_generation serve
.venv-stage2/bin/python -m stage2_prompt_generation serve --dry-run
```

두 명령 모두 `--config PATH`, `--vllm-executable PATH`를 지원한다.
상대 입력 경로는 프로젝트 루트 기준이며 출력 경로는 `outputs_stage2`로 고정한다.
`--limit`을 써도 전체 입력 검사는 서버 시작 전에 수행한다.

검수 실패는 결과 데이터이므로 전체 처리 완료 시 종료 코드는 0이다. 통신·실행 오류는 1,
Ctrl-C는 130이다. 통신 오류나 중단 시 재전송·재개 없이 종료하므로 그 실행은 24,000회에
도달하지 않을 수 있다. 이미 기록된 행은 남으며, 미완료 split의 요약은 생성하지 않는다.
기본 모드에서는 완료·오류·Ctrl-C 시 직접 시작한 서버를 종료한다.

`/v1/models`, `/version` 같은 서버 상태 확인은 LLM 생성 요청에 포함하지 않는다.

## 테스트

실제 LLM을 호출하지 않고, 12,000개 입력에 대한 정확한 24,000회 요청 수, 전체 생성 후 검수
순서, 실패 응답 보존, 행 단위 저장, JSON 형식, 집계, HTTP 재시도 부재를 검사한다.

```bash
.venv-stage2/bin/python -m unittest discover -s stage2_prompt_generation/tests -v
```
