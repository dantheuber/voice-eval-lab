"""A self-signed certificate for the LAN. Browsers only allow the mic on localhost or HTTPS."""

from __future__ import annotations

import datetime
import ipaddress
import logging
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

CERT_NAME = "voicelab.crt"
KEY_NAME = "voicelab.key"
_VALID_DAYS = 825  # the longest lifetime Apple platforms accept for a TLS certificate
_RENEW_BEFORE_DAYS = 14


def _covers(cert_path: Path, addresses: set[str]) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (OSError, ValueError, x509.ExtensionNotFound):
        return False
    held = {str(ip) for ip in names.get_values_for_type(x509.IPAddress)}
    margin = datetime.timedelta(days=_RENEW_BEFORE_DAYS)
    fresh = cert.not_valid_after_utc - margin > datetime.datetime.now(datetime.UTC)
    return fresh and addresses <= held


def ensure_certificate(directory: Path, lan_ip: str | None) -> tuple[Path, Path]:
    """Reuse the cert in `directory` unless it expired or the machine's LAN address changed."""
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / CERT_NAME, directory / KEY_NAME
    addresses = {"127.0.0.1", *([lan_ip] if lan_ip else [])}
    if cert_path.exists() and key_path.exists() and _covers(cert_path, addresses):
        return cert_path, key_path
    logger.info("generating a self-signed certificate for %s", sorted(addresses))
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "voicelab")])
    now = datetime.datetime.now(datetime.UTC)
    alternatives: list[x509.GeneralName] = [x509.DNSName("localhost")]
    alternatives += [x509.IPAddress(ipaddress.ip_address(a)) for a in sorted(addresses)]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=_VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.touch(mode=0o600)
    key_path.chmod(0o600)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path
