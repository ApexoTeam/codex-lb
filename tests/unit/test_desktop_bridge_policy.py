from __future__ import annotations

import pytest

from app.desktop_bridge import policy

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "target",
    [
        "/backend-api/codex",
        "/backend-api/codex/",
        "/backend-api/codex/future?q=%25%2e%2f",
    ],
)
def test_allowed_raw_target_is_preserved(target):
    assert policy.checked_target("PATCH", target) == target


@pytest.mark.parametrize(
    "target",
    [
        "*",
        "https://example.invalid/backend-api/codex/models",
        "/backend-api/codex/../secret",
        "/backend-api/codex/./models",
        "/backend-api/codex/%2E/models",
        "/backend-api/codex/%252e/models",
        "/backend-api/codex/%5Csecret",
        "/backend-api/codex/%2Fsecret",
        "/backend-api/codex\\secret",
        "/backend-api/codex//models",
        "/backend-api/codex/models#fragment",
        "/backend-api/codex-other",
    ],
)
def test_invalid_target_is_rejected(target):
    with pytest.raises(policy.PolicyRejection):
        policy.checked_target("GET", target)


@pytest.mark.parametrize("method", ["CONNECT", "TRACE"])
def test_tunnel_and_trace_methods_rejected(method):
    with pytest.raises(policy.PolicyRejection) as error:
        policy.checked_target(method, "/backend-api/codex/models")
    assert error.value.status == 405


@pytest.mark.parametrize(
    "hosts", [[], ["example.invalid:2456"], ["127.0.0.1:2455"], ["localhost:2456", "localhost:2456"]]
)
def test_host_is_exactly_one_local_authority(hosts):
    with pytest.raises(policy.PolicyRejection) as error:
        policy.check_host(hosts, 2456)
    assert error.value.status == 403


@pytest.mark.parametrize("value", ["localhost:2456", "127.0.0.1:2456"])
def test_local_host_is_admitted(value):
    policy.check_host([value], 2456)


def test_even_empty_origin_header_is_rejected():
    with pytest.raises(policy.PolicyRejection):
        policy.check_no_origin([""])


@pytest.mark.parametrize(
    "values", [[], ["Bearer x", "Bearer x"], ["Basic x"], ["Bearer"], ["Bearer  x"], ["Bearer x\n"]]
)
def test_bearer_rejections_use_403(values):
    with pytest.raises(policy.PolicyRejection) as error:
        policy.bearer_token(values)
    assert error.value.status == 403


def test_bearer_scheme_is_case_insensitive_but_token_is_exact():
    assert policy.bearer_token(["bEaReR synthetic"]) == "synthetic"
    assert policy.tokens_match("synthetic", "synthetic")
    assert not policy.tokens_match("SYNTHETIC", "synthetic")


def test_request_header_filter_removes_duplicates_and_connection_nominations():
    fields = [
        ("aUtHoRiZaTiOn", "secret"),
        ("AUTHORIZATION", "secret2"),
        ("X-OpenAI-Actor-Authorization", "actor"),
        ("ChatGPT-Account-ID", "account"),
        ("X-Forwarded-Future", "secret"),
        ("Connection", "X-Private, Accept-Encoding"),
        ("connection", "Another-Hop"),
        ("X-Private", "secret"),
        ("Another-Hop", "secret"),
        ("Accept-Encoding", "gzip"),
        ("User-Agent", "synthetic-client"),
        ("originator", "synthetic"),
        ("version", "0.0.0"),
        ("X-Codex-Future", "one"),
        ("X-Codex-Future", "two"),
    ]
    assert policy.request_headers(fields, "synthetic-lb", websocket=False) == [
        ("User-Agent", "synthetic-client"),
        ("originator", "synthetic"),
        ("version", "0.0.0"),
        ("X-Codex-Future", "one"),
        ("X-Codex-Future", "two"),
        ("Authorization", "Bearer synthetic-lb"),
    ]


def test_websocket_transport_headers_are_regenerated():
    fields = [
        ("Sec-WebSocket-Key", "key"),
        ("Sec-WebSocket-Version", "13"),
        ("Sec-WebSocket-Protocol", "protocol"),
        ("Sec-WebSocket-Extensions", "permessage-deflate"),
        ("Content-Length", "0"),
        ("X-Codex-Metadata", "preserved"),
    ]
    assert policy.request_headers(fields, "synthetic-lb", websocket=True) == [
        ("X-Codex-Metadata", "preserved"),
        ("Authorization", "Bearer synthetic-lb"),
    ]


def test_response_filters_credentials_without_collapsing_metadata():
    fields = [
        ("Set-Cookie", "secret"),
        ("set-cookie", "secret2"),
        ("WWW-Authenticate", "secret"),
        ("Proxy-Authenticate", "secret"),
        ("Authorization", "secret"),
        ("X-OpenAI-Actor-Authorization", "secret"),
        ("Connection", "X-Hop"),
        ("X-Hop", "secret"),
        ("X-Codex-Turn-State", "one"),
        ("X-Codex-Turn-State", "two"),
    ]
    assert policy.response_headers(fields, websocket=False) == [
        ("X-Codex-Turn-State", "one"),
        ("X-Codex-Turn-State", "two"),
    ]
