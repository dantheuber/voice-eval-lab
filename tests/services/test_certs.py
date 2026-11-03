from __future__ import annotations

import ipaddress
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from voicelab.services.certs import ensure_certificate


def names(path):
    cert = x509.load_pem_x509_certificate(path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return set(san.get_values_for_type(x509.DNSName)), set(san.get_values_for_type(x509.IPAddress))


def test_cert_names_localhost_and_the_lan_address(tmp_path):
    cert, _ = ensure_certificate(tmp_path, "192.168.1.50")
    dns, ips = names(cert)
    assert "localhost" in dns
    assert ipaddress.ip_address("192.168.1.50") in ips
    assert ipaddress.ip_address("127.0.0.1") in ips


def test_key_is_private_to_the_user(tmp_path):
    _, key = ensure_certificate(tmp_path, None)
    assert key.stat().st_mode & 0o077 == 0
    serialization.load_pem_private_key(key.read_bytes(), password=None)


def test_python_ssl_loads_the_pair(tmp_path):
    cert, key = ensure_certificate(tmp_path, "10.0.0.2")
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)


def test_existing_cert_is_reused(tmp_path):
    cert, _ = ensure_certificate(tmp_path, "192.168.1.50")
    before = cert.read_bytes()
    ensure_certificate(tmp_path, "192.168.1.50")
    assert cert.read_bytes() == before


def test_a_new_lan_address_regenerates_the_cert(tmp_path):
    cert, _ = ensure_certificate(tmp_path, "192.168.1.50")
    ensure_certificate(tmp_path, "192.168.1.51")
    assert ipaddress.ip_address("192.168.1.51") in names(cert)[1]


def test_expiring_cert_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr("voicelab.services.certs._VALID_DAYS", 1)  # inside the renewal margin
    cert, _ = ensure_certificate(tmp_path, None)
    before = cert.read_bytes()
    ensure_certificate(tmp_path, None)
    assert cert.read_bytes() != before


def test_a_corrupt_cert_is_replaced(tmp_path):
    cert, key = ensure_certificate(tmp_path, None)
    cert.write_text("garbage")
    ensure_certificate(tmp_path, None)
    x509.load_pem_x509_certificate(cert.read_bytes())
