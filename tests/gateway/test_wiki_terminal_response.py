from types import SimpleNamespace


def test_wiki_terminal_result_returns_exact_user_response(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/wiki")
    from gateway.run import _canonical_wiki_terminal_response

    response = (
        "internal verification\n"
        'INGEST RESULT {"status":"published","user_response":"[PUBLISHED] <https://example.test|open>"}\n'
        "prepare again"
    )
    assert _canonical_wiki_terminal_response(
        "slack", SimpleNamespace(profile="wiki"), response
    ) == "[PUBLISHED] <https://example.test|open>"


def test_nonterminal_or_nonwiki_response_is_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/wiki")
    from gateway.run import _canonical_wiki_terminal_response

    response = 'INGEST RESULT {"status":"failed","user_response":"nope"}'
    assert _canonical_wiki_terminal_response(
        "slack", SimpleNamespace(profile="wiki"), response
    ) == response
    assert _canonical_wiki_terminal_response(
        "telegram", SimpleNamespace(profile="wiki"), response
    ) == response


def test_duplicate_terminal_result_returns_its_exact_user_response(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/wiki")
    from gateway.run import _canonical_wiki_terminal_response

    response = 'INGEST RESULT {"status":"duplicate","user_response":"[DUPLICATE] exact"}'
    assert _canonical_wiki_terminal_response(
        "slack", SimpleNamespace(profile="wiki"), response
    ) == "[DUPLICATE] exact"
