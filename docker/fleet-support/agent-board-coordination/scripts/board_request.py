#!/usr/bin/env python3
"""Small dependency-free Agent Board HTTP client for skill workflows."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import shlex
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


BOARD_ENV_KEYS = {"BOARD_URL", "BOARD_AGENT_TOKEN"}
DEFAULT_BOARD_ENV_FILE = Path("/usr/local/etc/akh-medu/board.env")


def load_board_environment() -> None:
    """Load Board credentials when the daemon did not inherit them at startup."""
    env_file = Path(os.environ.get("BOARD_ENV_FILE", DEFAULT_BOARD_ENV_FILE))
    if not env_file.is_file():
        return
    try:
        contents = env_file.read_text()
    except PermissionError:
        return
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator or key not in BOARD_ENV_KEYS or key in os.environ:
            continue
        values = shlex.split(raw_value, comments=False, posix=True)
        if len(values) != 1:
            raise SystemExit(f"invalid {key} entry in {env_file}")
        os.environ[key] = values[0]


def read_json_argument(value: str) -> bytes:
    if value.startswith("@"):
        return Path(value[1:]).read_bytes()
    return value.encode()


def request(
    *,
    url: str,
    token: str,
    method: str,
    path: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(
        f"{url.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        payload = error.read()
        sys.stderr.write(payload.decode(errors="replace") + "\n")
        raise SystemExit(error.code) from error


def multipart(file_path: Path) -> tuple[str, bytes]:
    boundary = f"agent-board-{uuid.uuid4().hex}"
    if any(character in file_path.name for character in ('"', "\r", "\n")):
        raise SystemExit("upload filename contains an unsafe multipart character")
    media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode()
    body = prefix + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", body


def emit(headers: dict[str, str], body: bytes) -> None:
    content_type = headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        print(json.dumps(json.loads(body), indent=2))
    else:
        sys.stdout.buffer.write(body)
        if body and not body.endswith(b"\n"):
            print()


def header_value(headers: dict[str, str], name: str) -> str | None:
    """Return one HTTP field value without relying on field-name casing."""
    lowered_name = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == lowered_name), None
    )


def replay_thread(
    *, url: str, token: str, thread_id: str, limit: int, after_message_id: str | None
) -> None:
    """Stream a thread forward as newline-delimited JSON without page gaps."""
    normalized_thread_id = str(uuid.UUID(thread_id))
    cursor = str(uuid.UUID(after_message_id)) if after_message_id else None
    while True:
        parameters = {"limit": str(limit)}
        if cursor:
            parameters["after_message_id"] = cursor
        path = (
            f"/v1/threads/{normalized_thread_id}/messages?"
            f"{urllib.parse.urlencode(parameters)}"
        )
        _, headers, payload = request(
            url=url,
            token=token,
            method="GET",
            path=path,
        )
        messages = json.loads(payload)
        if not isinstance(messages, list):
            raise SystemExit("thread page was not a JSON message array")
        for message in messages:
            print(json.dumps(message, separators=(",", ":"), ensure_ascii=False))
        next_cursor = header_value(headers, "next-cursor")
        if next_cursor is None:
            return
        if not isinstance(next_cursor, str) or next_cursor == cursor:
            raise SystemExit("thread page returned an invalid or repeated next-cursor")
        cursor = str(uuid.UUID(next_cursor))


def main() -> None:
    load_board_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("BOARD_URL"))
    commands = parser.add_subparsers(dest="command", required=True)

    for method in ("get", "post", "delete"):
        command = commands.add_parser(method)
        command.add_argument("path")
        if method != "get":
            command.add_argument("data")

    upload = commands.add_parser("upload")
    upload.add_argument("message_id")
    upload.add_argument("file", type=Path)
    download = commands.add_parser("download")
    download.add_argument("attachment_id")
    download.add_argument("output", type=Path)
    replay = commands.add_parser(
        "replay-thread", help="stream a public thread forward as NDJSON"
    )
    replay.add_argument("thread_id")
    replay.add_argument("--after-message-id")
    replay.add_argument("--limit", type=int, default=200)

    args = parser.parse_args()
    token = os.environ.get("BOARD_AGENT_TOKEN")
    if not args.url or not token:
        parser.error("BOARD_URL and BOARD_AGENT_TOKEN are required")
    if args.command == "replay-thread" and not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")

    if args.command in {"get", "post", "delete"}:
        body = None if args.command == "get" else read_json_argument(args.data)
        _, headers, payload = request(
            url=args.url,
            token=token,
            method=args.command.upper(),
            path=args.path,
            body=body,
            content_type=None if body is None else "application/json",
        )
        emit(headers, payload)
    elif args.command == "replay-thread":
        replay_thread(
            url=args.url,
            token=token,
            thread_id=args.thread_id,
            limit=args.limit,
            after_message_id=args.after_message_id,
        )
    elif args.command == "upload":
        content_type, body = multipart(args.file)
        _, headers, payload = request(
            url=args.url,
            token=token,
            method="POST",
            path=f"/v1/messages/{args.message_id}/attachments",
            body=body,
            content_type=content_type,
        )
        emit(headers, payload)
    else:
        _, _, payload = request(
            url=args.url,
            token=token,
            method="GET",
            path=f"/v1/attachments/{args.attachment_id}",
        )
        args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
