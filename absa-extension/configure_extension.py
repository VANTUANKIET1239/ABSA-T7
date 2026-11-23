"""Point the unpacked extension at a tunnel without printing or packaging API credentials."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

APP = Path(__file__).resolve().parent


def configure(url, key_file=None):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {"", "/"}:
        raise ValueError("Use an API origin, without credentials, query or path.")
    if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Remote APIs require HTTPS.")
    origin = f"{parts.scheme}://{parts.netloc}"
    key = Path(key_file or APP / ".docker/api_key.txt").read_text(encoding="utf-8").strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Missing/invalid API key. Run prepare_docker.py first.")
    config = APP / "extension/api_config.local.json"
    config.write_text(json.dumps({"api_url": origin, "api_key": key}, indent=2) + "\n", encoding="utf-8")
    manifest_path = APP / "extension/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origins = list(dict.fromkeys(["http://127.0.0.1:8000", origin]))
    manifest["host_permissions"] = [host + "/*" for host in origins]
    manifest["content_security_policy"]["extension_pages"] = "script-src 'self'; object-src 'self'; connect-src 'self' " + " ".join(origins)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Extension API: {origin}")
    print("Private key saved locally (not printed, excluded from Git/ZIP). Reload the extension and product tab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url")
    target.add_argument("--from-tunnel", action="store_true")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    url = args.url
    if args.from_tunnel:
        logs = subprocess.run(["docker", "compose", "-f", str(APP / "compose.yaml"), "logs", "--no-color", "--tail", "200", "cloudflared"],
                              check=True, capture_output=True, text=True)
        matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", logs.stdout + logs.stderr)
        if not matches:
            raise RuntimeError("No tunnel URL found. Start the cloudflared service first.")
        url = matches[-1]
    configure(url, args.key_file)
