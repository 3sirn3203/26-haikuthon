# Stage3 이미지 생성

Stage2의 train/test 프롬프트를 받아 샘플마다 이미지 한 장과 원문 txt를 저장한다.
legacy의 vLLM-Omni 이미지 생성 경로를 옮긴 독립 패키지이며, 실행 시 legacy를 import하거나 읽지 않는다.
import만으로 서버 실행, 다운로드, 출력 저장을 수행하지 않는다.

## 입력과 출력

프로젝트 루트의 다음 두 파일을 입력으로 사용한다.

```text
outputs_stage2/train_prompts_v3.jsonl
outputs_stage2/test_prompts_v3.jsonl
```

각 행은 `id`, `prompt`를 포함한다. 기존 두 필드 형식과 Stage2의 새 형식을 모두 지원한다.
Train은 `group_id`까지 세 필드, test는 `group_id`, `cluster_id`, `is_query`까지 다섯 필드다.
메타데이터는 이미지 프롬프트에 추가하지 않으며, `is_query=false`도 이미지 후보로 처리한다.
입력 파일명은 `config.yaml`의 `prompt_files.train`, `prompt_files.test`로 선택한다.
이 키가 없는 기존 설정 파일은 이전 `train_prompts.jsonl`, `test_prompts.jsonl`을 사용한다.
현재 v3 입력은 train 4,000행, test 16,000행이다.
Test에는 이미지 후보 전체 16,000행을 입력한다. 평가 질의 8,000행만 담긴
`test_queries.jsonl`을 이미지 생성 입력으로 사용하지 않는다.

```json
{"id": "0000", "prompt": "Exactly two red cups on a white table."}
```

ID는 최소 네 자리 숫자 문자열이며 `10000`~`15999` 같은 다섯 자리 ID도 지원한다.
split 내부 중복은 허용하지 않으며, 번호가 빠져 있어도
원래 ID와 입력 순서를 유지한다. train과 test는 같은 ID를 사용할 수 있다.
빈 프롬프트, 잘못된 스키마, 빈 파일은 오류 처리한다. 요청한 모든 split의 전체 입력을
검증한 뒤 API를 호출한다. `--limit` 밖의 행도 입력 검증에 포함한다.

```text
outputs_stage3/
  train/
    0000/
      image.png
      prompt.txt
  test/
    0000/
      image.png
      prompt.txt
```

새 Stage2 결과의 `is_query=true` 행을 평가 질의로 선택하고 `test/*/image.png` 전체를 이미지 후보로 사용한다.
질의의 `id`와 이미지 폴더 이름이 정답을 연결한다. 기존 `test_queries.jsonl`을 사용하는 평가도 가능하다.
`prompt.txt`는 이미지 생성·재개용 원문이다.

`prompt.txt`는 실제 요청에 사용한 문자열을 UTF-8로 저장한다. 공백·줄바꿈을 바꾸거나
마지막에 줄바꿈을 추가하지 않는다. manifest, metadata, report, 로그, lock 파일은 만들지 않는다.
실행 중 저장용 임시 파일이 생길 수 있으며, 정상 종료 시 정리하고 다음 실행에서도 정리한다.

## 실행

Linux/NVIDIA CUDA GPU와 Python 3.12 이미지 전용 환경을 사용한다.
`requirements-image.txt`와 `requirements-image.lock.txt`는 legacy의 이미지용 파일과 동일하다.
vLLM과 vLLM-Omni는 모두 0.32.0이다. 실행에 사용할 Python 환경에 이 의존성이 준비되어 있어야 한다.
이미지 모델용 GPU를 확보한 뒤 프로젝트 루트에서 실행한다. Stage2 OpenAI Batch 실행에는 GPU가 필요하지 않다.

```bash
# 명령 확인만 수행: 모델 다운로드나 GPU 사용 없음
python -m stage3_image_generation serve --dry-run

# 터미널 1: 고정 revision의 base 모델·LoRA를 캐시에서 읽거나 다운로드한 뒤 서버 실행
python -m stage3_image_generation serve
```

서버의 Application startup complete 로그를 확인한 뒤 별도 터미널에서 실행한다.

```bash
# train/test 각 첫 한 개씩 생성
python -m stage3_image_generation generate --limit 1

# train 전체 생성 또는 재개
python -m stage3_image_generation generate --split train

# train 이후 test 전체 생성 또는 재개
python -m stage3_image_generation generate

# 선택한 범위만 재생성
python -m stage3_image_generation generate --split test --limit 4 --overwrite
```

두 명령 모두 `--config PATH`로 다른 YAML 설정을 사용할 수 있다. 설정 파일의 상대
`input_dir`와 `output_dir`는 프로젝트 루트를 기준으로 해석한다. 절대 경로도 사용할 수 있다.
서버와 클라이언트는 같은 설정으로 실행해야 하며, 모델·LoRA·서버 설정을 바꾸면 서버도 재시작한다.
클라이언트는 `/v1/models`의 모델 ID를 확인하지만 서버의 revision이나 LoRA를 별도로 증명하지 않는다.

## 기본 생성 설정

| 항목 | 값 |
| --- | --- |
| Base | `Qwen/Qwen-Image-2512`, BF16 |
| Base revision | `25468b98e3276ca6700de15c6628e51b7de54a26` |
| LoRA repository | `lightx2v/Qwen-Image-2512-Lightning` |
| LoRA revision | `4609161e308ce82018a42c2a0c9b107e865bb751` |
| LoRA file | `Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors` |
| LoRA backend / scale | `distill` / 1.0 |
| Steps / size | 8 / 768×768 |
| true CFG / negative prompt | 1.0 / 빈 문자열 |
| Server | `127.0.0.1:8091`, max sequences 4 |
| 실행 옵션 | step execution, CUDNN attention, VAE slicing, compile 활성 |
| Client | 동시 요청 4개, timeout 1,800초, 최대 2회 시도 |

`config.yaml`의 `generation.num_inference_steps`를 16으로 바꾸면 16 step으로 실행한다.
프롬프트는 수정하지 않으며, scene 기반 legacy seed 계산식의 식별자로 `train:0000` 또는
`test:0000`처럼 split과 ID를 함께 사용한다. base seed는 20261103이며, 실행 순서·재시도·
프롬프트 문구 변경은 같은 split/ID의 seed를 바꾸지 않는다. GPU·배치·패키지가 달라지면
픽셀 단위 재현성은 달라질 수 있다.

HTTP 429·5xx와 연결/timeout 오류는 legacy처럼 2초 뒤 재시도한다. 다른 HTTP 4xx는 즉시 실패한다.
PNG 형식·크기·파일 무결성을 검사하며, 이미지 내용의 의미나 품질은 검수하지 않는다.

## 재개와 중단

- 정상 PNG와 입력에 정확히 일치하는 `prompt.txt`가 있으면 서버 호출 없이 재사용한다.
- 파일이 없거나 PNG가 손상되었거나 크기가 다르면 해당 샘플을 다시 생성한다.
- 기존 txt가 입력과 다르면 생성 전에 오류 처리한다. 의도한 교체는 `--overwrite`를 사용한다.
- `--overwrite`는 선택한 split·limit 범위만 덮어쓰며 다른 샘플은 삭제하지 않는다.
- **이력이 없으므로 모델·LoRA·step·CFG·seed 등 설정 변경은 감지하지 못한다.** 설정을 바꾼
  뒤에는 `--overwrite`를 사용하거나 YAML의 `output_dir`를 새로운 경로로 지정한다.
- 이미지 교체 전에 기존 txt 완료 표식을 지우고, 새로운 txt는 마지막에 교체한다.
  저장이 중단되면 다음 실행에서 미완료 샘플을 재생성한다.
- 출력 디렉터리 자체에 프로세스 잠금을 걸어 같은 출력 경로의 동시 실행을 차단한다.
  첫 Ctrl-C에서는 새 요청 제출을 멈추고 진행 중 요청이 저장될 때까지 기다린다.
- 오류와 수량은 콘솔에 출력한다. 성공은 종료 코드 0, 실패가 남으면 1, Ctrl-C는 130이다.

## 기본 문법 점검

프로젝트 루트에서 의존성 import나 추론 없이 Python 소스를 파싱할 수 있다.

```bash
python - <<'PY'
import ast
from pathlib import Path
for path in sorted(Path('stage3_image_generation').glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print('OK', path)
PY
```
