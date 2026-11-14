"""Self-signed TLS certificate for the VR server.

WebXR (immersive VR in the Meta Quest browser) only works in a *secure context*, i.e. https:// (or localhost).
We generate a self-signed certificate once; the first time, the Quest browser shows a warning that you accept
with "Advanced -> Proceed".
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import socket
from pathlib import Path


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:  # primary outbound interface (no packets are sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def ensure_cert(certs_dir: str | Path) -> tuple[Path, Path]:
    """Return (cert.pem, key.pem), creating or refreshing them when the machine's IPs changed."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    d = Path(certs_dir)
    d.mkdir(parents=True, exist_ok=True)
    cert_p, key_p, meta_p = d / "cert.pem", d / "key.pem", d / "cert.json"
    host = socket.gethostname()
    ips = local_ips()
    if cert_p.exists() and key_p.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if set(ips) <= set(meta.get("ips", [])) and meta.get("host") == host:
                return cert_p, key_p
        except ValueError:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{host} posture coach")])
    sans = [x509.DNSName(n) for n in {host, f"{host}.local", "localhost"}]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=3 * 365))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    key_p.chmod(0o600)
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    meta_p.write_text(json.dumps({"host": host, "ips": ips}), encoding="utf-8")
    return cert_p, key_p
