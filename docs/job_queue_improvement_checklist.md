# Supabase PostgreSQL SKIP LOCKED 큐 구현 체크리스트

## 배경
- Ngrok latency 13099ms 경고 발생
- 여러 오디오 파일 동시 업로드 시 BackgroundTasks 병목
- STT 처리 서버 2대를 효율적으로 활용하기 위한 작업 큐 필요

## 1. 데이터베이스 스키마 준비

- [ ] **작업 큐 테이블 생성** (`job_queue` 또는 기존 `sessions` 테이블 활용)
  - `id` (UUID, PK)
  - `session_id` (FK to sessions)
  - `status` (pending / processing / completed / failed)
  - `payload` (JSONB - s3_key, stt_model, user_id 등)
  - `created_at`, `started_at`, `completed_at`
  - `worker_id` (어떤 worker가 처리 중인지)
  - `retry_count`, `max_retries`
  - `error_message` (실패 시)

- [ ] **인덱스 생성** (폴링 성능 최적화)
  ```sql
  CREATE INDEX idx_job_queue_pending ON job_queue(status, created_at) 
  WHERE status = 'pending';
  ```

## 2. 백엔드 코드 구현

- [ ] **Job Queue 서비스 모듈 생성** (`backend/services/job_queue.py`)
  - `enqueue_job()` - 작업 추가
  - `fetch_next_job()` - SKIP LOCKED로 다음 작업 가져오기
  - `complete_job()` - 성공 처리
  - `fail_job()` - 실패 처리 (재시도 로직 포함)

- [ ] **Worker 모듈 생성** (`backend/worker/stt_worker.py`)
  - 폴링 루프 구현
  - Graceful shutdown 처리
  - Worker ID 생성 (어떤 인스턴스가 처리 중인지 추적)

- [ ] **기존 API 수정** (`session_controller.py`)
  - `BackgroundTasks` 대신 `enqueue_job()` 호출
  - 즉시 202 응답 반환

## 3. Worker 실행 환경

- [ ] **Worker 실행 스크립트** (`run_worker.sh`)
  - STT 서버 2대에서 각각 worker 실행

- [ ] **환경 변수 설정**
  - `WORKER_POLL_INTERVAL` (폴링 주기, 예: 5초)
  - `WORKER_ID` (서버 식별자)
  - `MAX_RETRIES` (재시도 횟수)

## 4. 모니터링 & 관리

- [ ] **상태 확인 API** (선택사항)
  - 현재 큐 길이 조회
  - 처리 중인 작업 목록
  - 실패한 작업 목록

- [ ] **Stale job 처리**
  - Worker가 죽었을 때 `processing` 상태로 멈춘 작업 복구
  - 예: 30분 이상 `processing`이면 `pending`으로 되돌리기

## 5. 테스트

- [ ] 단일 작업 처리 테스트
- [ ] 동시 다중 작업 처리 테스트
- [ ] Worker 중단 시 작업 복구 테스트
- [ ] 재시도 로직 테스트

## 구현 순서 권장

```
1단계: DB 스키마 (SQL 실행)
    ↓
2단계: job_queue 서비스 모듈
    ↓
3단계: session_controller 수정
    ↓
4단계: Worker 구현
    ↓
5단계: 테스트 & 배포
```

## 기대 효과

| 지표          | 현재 (BackgroundTasks) | 개선 후 (Job Queue) |
| ------------- | ---------------------- | ------------------- |
| API 응답 시간 | 수 초~수십 초          | < 100ms             |
| Ngrok latency | 13099ms (경고)         | < 1000ms            |
| 동시 처리     | 단일 프로세스          | Worker 수만큼 병렬  |
| 장애 복구     | 불가                   | 자동 재시도         |
