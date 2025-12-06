#!/usr/bin/env python3
"""
성능 테스트 스크립트 - 현재 BackgroundTasks 기반 시스템 베이스라인 측정

테스트 항목:
1. 단일 요청 응답 시간
2. 동시 다중 요청 응답 시간
3. 서버 부하 시 응답 시간 변화

사용법:
    python tests/performance/test_baseline.py --url <API_URL> --count <요청수>
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import httpx


@dataclass
class RequestResult:
    """단일 요청 결과"""

    request_id: int
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class TestReport:
    """테스트 결과 리포트"""

    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    min_response_time_ms: float
    max_response_time_ms: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    total_test_duration_s: float
    requests_per_second: float
    results: List[RequestResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": f"{(self.successful_requests / self.total_requests * 100):.2f}%",
            },
            "response_times_ms": {
                "min": round(self.min_response_time_ms, 2),
                "max": round(self.max_response_time_ms, 2),
                "avg": round(self.avg_response_time_ms, 2),
                "median": round(self.median_response_time_ms, 2),
                "p95": round(self.p95_response_time_ms, 2),
                "p99": round(self.p99_response_time_ms, 2),
            },
            "throughput": {
                "total_duration_s": round(self.total_test_duration_s, 2),
                "requests_per_second": round(self.requests_per_second, 2),
            },
        }

    def print_report(self):
        print("\n" + "=" * 60)
        print(f"📊 테스트 결과: {self.test_name}")
        print("=" * 60)
        print(f"\n📈 요청 통계:")
        print(f"   총 요청 수: {self.total_requests}")
        print(f"   성공: {self.successful_requests}")
        print(f"   실패: {self.failed_requests}")
        print(
            f"   성공률: {(self.successful_requests / self.total_requests * 100):.2f}%"
        )
        print(f"\n⏱️  응답 시간 (ms):")
        print(f"   최소: {self.min_response_time_ms:.2f}")
        print(f"   최대: {self.max_response_time_ms:.2f}")
        print(f"   평균: {self.avg_response_time_ms:.2f}")
        print(f"   중앙값: {self.median_response_time_ms:.2f}")
        print(f"   P95: {self.p95_response_time_ms:.2f}")
        print(f"   P99: {self.p99_response_time_ms:.2f}")
        print(f"\n🚀 처리량:")
        print(f"   총 테스트 시간: {self.total_test_duration_s:.2f}초")
        print(f"   초당 요청 수: {self.requests_per_second:.2f}")
        print("=" * 60 + "\n")


def percentile(data: List[float], p: float) -> float:
    """백분위수 계산"""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def generate_test_payload(request_id: int) -> dict:
    """테스트용 요청 페이로드 생성"""
    return {
        "user_id": 698,
        "title": f"성능테스트_{request_id}_{int(time.time())}",
        "s3_key": "audio/698/1764833515657_3a5c25bf-7596-4e93-9d48-6e490b345e1f",  # 테스트용 S3 키
        "file_size_mb": 7.6,
        "duration_seconds": 300.0,
        "stt_model": "whisper",
        "template_id": 1,
    }


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    request_id: int,
    payload: dict,
) -> RequestResult:
    """단일 HTTP 요청 전송 및 측정"""
    start_time = time.perf_counter()

    try:
        response = await client.post(
            url,
            json=payload,
            timeout=120.0,  # 2분 타임아웃
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        session_id = None
        if response.status_code in (200, 201, 202):
            try:
                data = response.json()
                session_id = data.get("session_id")
            except:
                pass

        return RequestResult(
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=elapsed_ms,
            success=response.status_code in (200, 201, 202),
            session_id=session_id,
        )
    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            request_id=request_id,
            status_code=0,
            response_time_ms=elapsed_ms,
            success=False,
            error_message="Timeout",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            request_id=request_id,
            status_code=0,
            response_time_ms=elapsed_ms,
            success=False,
            error_message=str(e),
        )


def analyze_results(
    test_name: str, results: List[RequestResult], duration: float
) -> TestReport:
    """결과 분석 및 리포트 생성"""
    response_times = [r.response_time_ms for r in results]
    successful = [r for r in results if r.success]

    return TestReport(
        test_name=test_name,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(results) - len(successful),
        min_response_time_ms=min(response_times) if response_times else 0,
        max_response_time_ms=max(response_times) if response_times else 0,
        avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
        median_response_time_ms=(
            statistics.median(response_times) if response_times else 0
        ),
        p95_response_time_ms=percentile(response_times, 95) if response_times else 0,
        p99_response_time_ms=percentile(response_times, 99) if response_times else 0,
        total_test_duration_s=duration,
        requests_per_second=len(results) / duration if duration > 0 else 0,
        results=results,
    )


async def test_sequential(url: str, count: int) -> TestReport:
    """순차 요청 테스트 - 한 번에 하나씩"""
    print(f"\n🔄 순차 요청 테스트 시작 ({count}개 요청)...")

    results = []
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for i in range(count):
            payload = generate_test_payload(i)
            result = await send_request(client, url, i, payload)
            results.append(result)
            print(
                f"   요청 {i+1}/{count}: {result.response_time_ms:.2f}ms - {'✅' if result.success else '❌'}"
            )

    duration = time.perf_counter() - start_time
    return analyze_results("순차 요청 테스트 (Sequential)", results, duration)


async def test_concurrent(url: str, count: int) -> TestReport:
    """동시 요청 테스트 - 모든 요청을 동시에"""
    print(f"\n⚡ 동시 요청 테스트 시작 ({count}개 요청 동시 전송)...")

    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            send_request(client, url, i, generate_test_payload(i)) for i in range(count)
        ]
        results = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time

    for i, result in enumerate(results):
        print(
            f"   요청 {i+1}: {result.response_time_ms:.2f}ms - {'✅' if result.success else '❌'}"
        )

    return analyze_results("동시 요청 테스트 (Concurrent)", list(results), duration)


async def test_batch(url: str, total: int, batch_size: int) -> TestReport:
    """배치 요청 테스트 - batch_size개씩 동시에"""
    print(f"\n📦 배치 요청 테스트 시작 (총 {total}개, 배치당 {batch_size}개)...")

    results = []
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_num = batch_start // batch_size + 1
            print(f"   배치 {batch_num} 전송 중 ({batch_start+1}~{batch_end})...")

            tasks = [
                send_request(client, url, i, generate_test_payload(i))
                for i in range(batch_start, batch_end)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # 배치 간 짧은 대기
            await asyncio.sleep(0.1)

    duration = time.perf_counter() - start_time
    return analyze_results(
        f"배치 요청 테스트 (Batch size={batch_size})", results, duration
    )


async def health_check(url: str) -> bool:
    """서버 상태 확인"""
    try:
        # /api/v2/session 의 base URL에서 health check
        base_url = url.rsplit("/api", 1)[0]
        health_url = f"{base_url}/health"

        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=10.0)
            return response.status_code == 200
    except Exception as e:
        print(f"⚠️  Health check 실패: {e}")
        return False


async def run_all_tests(url: str, count: int, output_file: Optional[str] = None):
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("🚀 성능 테스트 시작")
    print(f"   대상 URL: {url}")
    print(f"   요청 수: {count}")
    print("=" * 60)

    # Health check
    print("\n🏥 서버 상태 확인 중...")
    if not await health_check(url):
        print("⚠️  서버 Health check 실패 - 테스트를 계속 진행합니다.")
    else:
        print("✅ 서버 정상")

    reports = []

    # 1. 순차 요청 테스트
    report1 = await test_sequential(url, min(count, 5))  # 순차는 5개로 제한
    report1.print_report()
    reports.append(report1)

    await asyncio.sleep(2)  # 서버 안정화 대기

    # 2. 동시 요청 테스트
    report2 = await test_concurrent(url, count)
    report2.print_report()
    reports.append(report2)

    await asyncio.sleep(2)

    # 3. 배치 요청 테스트 (3개씩)
    if count >= 6:
        report3 = await test_batch(url, count, 3)
        report3.print_report()
        reports.append(report3)

    # 결과 저장
    if output_file:
        output_data = {
            "test_config": {
                "url": url,
                "count": count,
                "timestamp": datetime.now().isoformat(),
            },
            "reports": [r.to_dict() for r in reports],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 {output_file}에 저장되었습니다.")

    # 요약
    print("\n" + "=" * 60)
    print("📋 테스트 요약")
    print("=" * 60)
    for report in reports:
        print(f"\n{report.test_name}:")
        print(f"   평균 응답시간: {report.avg_response_time_ms:.2f}ms")
        print(f"   P95 응답시간: {report.p95_response_time_ms:.2f}ms")
        print(
            f"   성공률: {(report.successful_requests / report.total_requests * 100):.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="세션 API 성능 테스트")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:25500/api/v2/session",
        help="테스트할 API URL (기본값: http://localhost:25500/api/v2/session)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="테스트 요청 수 (기본값: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 요청 없이 설정만 확인",
    )

    args = parser.parse_args()

    if args.dry_run:
        print(f"URL: {args.url}")
        print(f"Count: {args.count}")
        print(f"Output: {args.output}")
        print("Dry run 완료")
        return

    asyncio.run(run_all_tests(args.url, args.count, args.output))


if __name__ == "__main__":
    main()
