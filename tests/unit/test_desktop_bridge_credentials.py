from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.desktop_bridge import credentials

pytestmark = pytest.mark.unit


def private_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def test_credential_repr_never_contains_tokens():
    value = credentials.LocalCredentials("synthetic-client-secret", "synthetic-lb-secret")
    assert "synthetic-client-secret" not in repr(value)
    assert "synthetic-lb-secret" not in repr(value)


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r\n"])
def test_lb_token_accepts_one_optional_line_ending(tmp_path, ending):
    path = private_file(tmp_path / "token", b"synthetic-token" + ending)
    before = path.read_bytes()
    assert credentials.read_lb_token(path) == "synthetic-token"
    assert path.read_bytes() == before


@pytest.mark.parametrize("content", [b"", b" token", b"token ", b"one\ntwo", b"one\n\n", b"one\r", b"one\t", b"\xff"])
def test_lb_token_rejects_unsafe_header_values(tmp_path, content):
    path = private_file(tmp_path / "token", content)
    with pytest.raises(credentials.CredentialError):
        credentials.read_lb_token(path)


@pytest.mark.parametrize("extra", [{}, {"auth_mode": "chatgpt"}, {"OPENAI_API_KEY": None}, {"OPENAI_API_KEY": ""}])
def test_chatgpt_auth_accepts_current_and_older_files(tmp_path, extra):
    path = private_file(
        tmp_path / "auth.json", json.dumps({"tokens": {"access_token": "synthetic-token"}, **extra}).encode()
    )
    before = path.read_bytes()
    assert credentials.read_access_token(tmp_path) == "synthetic-token"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b"[]",
        b"null",
        b"{}",
        b'{"tokens":[]}',
        b'{"tokens":{"access_token":42}}',
        b'{"tokens":{"access_token":""}}',
        b'{"tokens":{"access_token":"token\\nInjected"}}',
        b'{"auth_mode":"apikey","tokens":{"access_token":"synthetic-token"}}',
        b'{"OPENAI_API_KEY":"synthetic-key","tokens":{"access_token":"synthetic-token"}}',
    ],
)
def test_auth_file_rejects_malformed_or_unsupported_credentials(tmp_path, content):
    path = private_file(tmp_path / "auth.json", content)
    with pytest.raises(credentials.CredentialError) as error:
        credentials.read_access_token(tmp_path)
    assert str(path) not in str(error.value)
    assert "synthetic-token" not in str(error.value)
    assert "synthetic-key" not in str(error.value)


def test_file_size_is_bounded(tmp_path):
    path = private_file(tmp_path / "token", b"x" * (credentials.MAX_TOKEN_FILE_BYTES + 1))
    with pytest.raises(credentials.CredentialError, match="too large"):
        credentials.read_lb_token(path)


def test_missing_file_error_omits_path(tmp_path):
    path = tmp_path / "private-directory-name" / "token"
    with pytest.raises(credentials.CredentialError) as error:
        credentials.read_lb_token(path)
    assert str(path) not in str(error.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership/mode contract")
@pytest.mark.parametrize("mode", [0o640, 0o604, 0o601, 0o620])
def test_group_or_other_permissions_rejected(tmp_path, mode):
    path = private_file(tmp_path / "token", b"synthetic-token")
    path.chmod(mode)
    with pytest.raises(credentials.CredentialError, match="chmod 600"):
        credentials.read_lb_token(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership/mode contract")
def test_foreign_owner_is_rejected(tmp_path, monkeypatch):
    path = private_file(tmp_path / "token", b"synthetic-token")
    monkeypatch.setattr(credentials, "_CURRENT_UID", lambda: os.getuid() + 1)
    with pytest.raises(credentials.CredentialError, match="owned by you"):
        credentials.read_lb_token(path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires no-follow file open")
def test_final_symlink_is_rejected(tmp_path):
    real = private_file(tmp_path / "real", b"synthetic-token")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(credentials.CredentialError):
        credentials.read_lb_token(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO")
def test_fifo_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(credentials.CredentialError, match="regular file"):
        credentials.read_lb_token(fifo)
