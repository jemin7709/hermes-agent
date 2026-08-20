from types import SimpleNamespace


def test_terminal_tool_result_keeps_exact_response_and_ignores_failed():
    from agent.tool_executor import _capture_wiki_terminal_response

    agent = SimpleNamespace(_wiki_terminal_guard=True)
    assert _capture_wiki_terminal_response(
        agent,
        'INGEST RESULT {"status":"published","user_response":"done"}',
    )
    assert agent._wiki_terminal_response == "done"
    assert not _capture_wiki_terminal_response(
        agent,
        'INGEST RESULT {"status":"failed","user_response":"nope"}',
    )
    assert agent._wiki_terminal_response == "done"


def test_non_wiki_result_is_unchanged():
    from agent.tool_executor import _capture_wiki_terminal_response

    agent = SimpleNamespace(_wiki_terminal_guard=False)
    assert not _capture_wiki_terminal_response(
        agent,
        'INGEST RESULT {"status":"duplicate","user_response":"already"}',
    )
    assert not hasattr(agent, "_wiki_terminal_response")


def test_wiki_terminal_seam_stops_before_a_second_tool():
    from agent.tool_executor import _capture_wiki_terminal_response

    agent = SimpleNamespace(_wiki_terminal_guard=True)
    calls = []
    first = 'INGEST RESULT {"status":"published","user_response":"done"}'
    calls.append("prepare")
    if not _capture_wiki_terminal_response(agent, first):
        calls.append("second_prepare")
    assert agent._wiki_terminal_response == "done"
    assert calls == ["prepare"]