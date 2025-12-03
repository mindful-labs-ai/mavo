from fastapi import HTTPException, status
from util.logger import get_logger
from backend.config import S3_ACCESS_KEY, S3_BUCKET_NAME, S3_REGION, S3_SECRET_KEY
import boto3
from botocore.config import Config


logger = get_logger(__name__)


async def issue_presigned_url(s3_key: str) -> str:
    """
    S3 presigned URL을 발급 (유효기간 10분).
    """

    if not all([S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 환경변수(S3_REGION, S3_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME)가 설정되지 않았습니다.",
        )

    try:
        s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=600,
        )
        return presigned_url
    except Exception as e:
        logger.error(f"Presigned URL 발급 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 presigned URL 발급에 실패했습니다.",
        )
