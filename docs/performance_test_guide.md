# 성능 테스트 실행 가이드

## 빠른 시작

```bash
# 로컬 서버 테스트
python tests/performance/test_baseline.py --count 5

# Ngrok 서버 테스트
python tests/performance/test_baseline.py \
  --url https://your-ngrok-url.ngrok.io/api/v2/session \
  --count 10
```

## 옵션

| 옵션        | 설명                       | 기본값                                  |
| ----------- | -------------------------- | --------------------------------------- |
| `--url`     | API 엔드포인트 URL         | `http://localhost:25500/api/v2/session` |
| `--count`   | 테스트 요청 수             | 5                                       |
| `--output`  | 결과 저장 파일 (JSON)      | 없음                                    |
| `--dry-run` | 설정 확인만 (요청 안 보냄) | -                                       |

## 테스트 종류

1. **순차 요청** - 하나씩 순서대로 (최대 5개)
2. **동시 요청** - 모든 요청 동시 전송
3. **배치 요청** - 3개씩 묶어서 전송 (6개 이상일 때)

## 예시

```bash
# 결과를 파일로 저장
python tests/performance/test_baseline.py \
  --url https://abc123.ngrok.io/api/v2/session \
  --count 10 \
  --output docs/performance_result.json

# 설정만 확인
python tests/performance/test_baseline.py --dry-run
```

## 출력 예시

```
📊 테스트 결과: 동시 요청 테스트 (Concurrent)
============================================================
📈 요청 통계:
   총 요청 수: 10
   성공: 10
   실패: 0
   성공률: 100.00%

⏱️  응답 시간 (ms):
   평균: 245.32
   P95: 512.45
   P99: 890.12
```

## 개선 전/후 비교

```bash
# 1. 개선 전 베이스라인 측정
python tests/performance/test_baseline.py \
  --output docs/baseline_before.json

# 2. Job Queue 구현 후 다시 측정
python tests/performance/test_baseline.py \
  --output docs/baseline_after.json
```
