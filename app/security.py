"""HMAC-SHA256 签名令牌。

同一套机制服务两种身份，客户端只能持有服务端签发的 (payload, signature) 对：

  - sub="access"      全局访问令牌，由 /api/auth 签发
  - sub=<patient_id>  患者身份令牌，由注册/登录签发

任何对载荷的篡改（例如把 sub 改成别人的 patient_id）都会导致验签失败，
这替代了此前"信任客户端送来的 x-patient-id 头"的做法。
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

ACCESS_SUBJECT = "access"
ACCESS_TTL = 12 * 3600          # 全局访问令牌 12 小时
PATIENT_TTL = 30 * 24 * 3600    # 患者令牌 30 天


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret() -> bytes:
    """签名密钥：优先 SECRET_KEY，否则从 ACCESS_PASSWORD 派生。

    派生是为了不强制新增配置项；两者任一变更都会让已签发的令牌立即失效。
    """
    raw = os.getenv("SECRET_KEY", "")
    if raw:
        return hashlib.sha256(raw.encode("utf-8")).digest()
    password = os.getenv("ACCESS_PASSWORD", "")
    return hashlib.sha256(f"agentic-rag:{password}".encode("utf-8")).digest()


def _sign(payload: bytes) -> str:
    return _b64e(hmac.new(_secret(), payload, hashlib.sha256).digest())


def issue_token(subject: str, ttl: int) -> str:
    """签发令牌，格式 ``payload.signature``（均为 base64url）。"""
    now = int(time.time())
    payload = json.dumps(
        {"sub": subject, "iat": now, "exp": now + ttl},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{_b64e(payload)}.{_sign(payload)}"


def verify_token(token: str) -> Optional[dict]:
    """验签并校验有效期。任何失败都返回 None，由调用方转成 401。"""
    if not token or "." not in token:
        return None

    payload_b64, signature = token.rsplit(".", 1)
    try:
        payload = _b64d(payload_b64)
    except (ValueError, TypeError):
        return None

    # 用收到的载荷字节重算签名，篡改载荷必然导致不一致
    if not hmac.compare_digest(_sign(payload), signature):
        return None

    try:
        claims = json.loads(payload)
    except ValueError:
        return None

    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if not isinstance(exp, int) or exp < time.time():
        return None
    return claims
