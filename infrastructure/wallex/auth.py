from __future__ import annotations

import hashlib
import hmac
import time
from typing import Dict, Mapping, Optional


def sign_hmac_sha256(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical_params(params: Optional[Mapping[str, object]]) -> str:
    if not params:
        return ""
    # sort keys for deterministic signing; stringify values
    items = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "&".join(items)


def auth_headers(
    api_key: str,
    api_secret: str,
    payload: Dict[str, object] | None = None,
    header_names: Optional[Dict[str, str]] = None,
    ts: Optional[str] = None,
) -> Dict[str, str]:
    """Legacy/simple signing: sign only timestamp + payload params.
    Some Wallex endpoints may require method/path in the signature; use wallex_signed_headers for that if needed.
    """
    ts = ts or str(int(time.time() * 1000))
    payload_str = _canonical_params(payload)
    to_sign = f"timestamp={ts}{('&' + payload_str) if payload_str else ''}"
    signature = sign_hmac_sha256(api_secret, to_sign)
    hn = header_names or {"key": "X-API-KEY", "sign": "X-API-SIGN", "ts": "X-API-TS"}
    return {
        hn["key"]: api_key,
        hn["sign"]: signature,
        hn["ts"]: ts,
        "Content-Type": "application/json",
    }


def wallex_signed_headers(
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
    params: Optional[Mapping[str, object]] = None,
    body: Optional[Mapping[str, object]] = None,
    header_names: Optional[Dict[str, str]] = None,
    ts: Optional[str] = None,
) -> Dict[str, str]:
    """More complete signing: include HTTP method and path in message.
    Adjust this canonical form to match the official Wallex spec from the PDF if it differs.
    """
    ts = ts or str(int(time.time() * 1000))
    p = _canonical_params(params)
    b = _canonical_params(body)
    base = f"timestamp={ts}&method={method.upper()}&path={path}"
    if p:
        base += f"&params={p}"
    if b:
        base += f"&body={b}"
    signature = sign_hmac_sha256(api_secret, base)
    hn = header_names or {"key": "X-API-KEY", "sign": "X-API-SIGN", "ts": "X-API-TS"}
    return {
        hn["key"]: api_key,
        hn["sign"]: signature,
        hn["ts"]: ts,
        "Content-Type": "application/json",
    }
