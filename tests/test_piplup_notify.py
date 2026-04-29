"""Tests for optional piplup notification delivery."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Queue
from threading import Thread

from piplup_notify import PiplupNotificationConfig, PiplupNotifier


def test_notifier_is_noop_without_token(capsys) -> None:
    """Missing notification credentials should not break a simulation run."""
    config = PiplupNotificationConfig(enabled=True, token=None, run_id="run-no-token")
    notifier = PiplupNotifier(config)

    assert notifier.notify_started({"backend": "analytic"}) is False

    captured = capsys.readouterr()
    assert "token is not configured" in captured.out


def test_notifier_posts_normalized_event() -> None:
    """Enabled notifications should POST normalized events to /api/events."""
    received: Queue[dict[str, object]] = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.put(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(body.decode("utf-8")),
                }
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = PiplupNotificationConfig(
            enabled=True,
            base_url=f"http://127.0.0.1:{server.server_port}",
            token="secret-token",
            account=None,
            run_id="run-123",
            timeout_s=1.0,
        )
        notifier = PiplupNotifier(config)

        assert notifier.notify_finished({"summary": "done", "measurements": 3}) is True
        record = received.get(timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert record["path"] == "/api/events"
    assert record["authorization"] == "Bearer secret-token"
    body = record["body"]
    assert body["source"] == "rotating-shield-pf"
    assert body["type"] == "simulation.finished"
    assert body["severity"] == "info"
    assert body["dedupe_key"] == "rotating-shield-pf:run-123:finished"
    assert "account" not in body
    assert body["payload"]["run_id"] == "run-123"
    assert body["payload"]["status"] == "succeeded"
    assert body["payload"]["summary"] == "done"


def test_notifier_posts_spectrum_event_with_step_dedupe() -> None:
    """Spectrum notifications should use per-step dedupe keys."""
    received: Queue[dict[str, object]] = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.put(json.loads(body.decode("utf-8")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = PiplupNotificationConfig(
            enabled=True,
            base_url=f"http://127.0.0.1:{server.server_port}",
            token="secret-token",
            account=None,
            run_id="run-123",
            timeout_s=1.0,
        )
        notifier = PiplupNotifier(config)

        assert notifier.notify_spectrum(7, {"energy_keV": [662.0]}) is True
        body = received.get(timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert body["type"] == "simulation.spectrum"
    assert body["id"] == "rotating-shield-pf-run-123-spectrum-0007"
    assert body["dedupe_key"] == "rotating-shield-pf:run-123:spectrum-0007"
    assert body["payload"]["step_index"] == 7
    assert body["payload"]["energy_keV"] == [662.0]


def test_config_from_env_is_opt_in(monkeypatch) -> None:
    """Environment configuration should keep notifications disabled by default."""
    monkeypatch.delenv("PIPLUP_NOTIFY_ENABLED", raising=False)
    monkeypatch.setenv("PIPLUP_NOTIFY_TOKEN", "env-token")

    disabled = PiplupNotificationConfig.from_env()
    enabled = PiplupNotificationConfig.from_env(enabled=True, run_id="env-run")

    assert disabled.enabled is False
    assert disabled.token == "env-token"
    assert enabled.enabled is True
    assert enabled.token == "env-token"
    assert enabled.run_id == "env-run"
