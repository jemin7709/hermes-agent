import asyncio
import concurrent.futures
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_constants
import hermes_cli.profiles as profiles
import gateway.run as run_module
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from plugins.platforms.slack.adapter import SlackAdapter


def _wiki_identity(monkeypatch, tmp_path):
    home = tmp_path / "wiki"
    home.mkdir()
    monkeypatch.setattr(run_module, "_hermes_home", home)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: home)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "wiki")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    return home


def _live_runner_source(*, source_profile=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"notice_delivery": "private"},
            )
        }
    )
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.typed_command_prefix = "/"
    adapter.pause_typing_for_chat = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="public"))
    adapter.send_exec_approval = AsyncMock(
        return_value=SendResult(success=True, message_id="public-approval")
    )
    adapter.send_private_notice = AsyncMock(
        return_value=SendResult(success=True, message_id="private")
    )
    runner.adapters = {Platform.SLACK: adapter}
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        user_id="U123",
        thread_id="111.222",
        profile=source_profile,
    )
    source._transport_adapter_ref = weakref.ref(adapter)
    runner._thread_metadata_for_source = MagicMock(
        return_value={"thread_id": "111.222"}
    )
    return runner, adapter, source


def test_wiki_requires_live_registered_transport_and_private_override(monkeypatch, tmp_path):
    _wiki_identity(monkeypatch, tmp_path)
    runner, adapter, source = _live_runner_source()

    assert run_module._wiki_private_slack_adapter_for_source(runner, source) is adapter

    source_without_provenance = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        user_id="U123",
        profile="wiki",
    )
    assert (
        run_module._wiki_private_slack_adapter_for_source(
            runner, source_without_provenance
        )
        is None
    )

    source.profile_route_rejected = True
    assert run_module._wiki_private_slack_adapter_for_source(runner, source) is None


def test_wiki_identity_matrix_rejects_stale_home_multiplex_spoof_and_base_fallback(
    monkeypatch, tmp_path
):
    home = _wiki_identity(monkeypatch, tmp_path)
    runner, adapter, source = _live_runner_source(source_profile="wiki")

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home / "stale")
    assert run_module._wiki_private_slack_adapter_for_source(runner, source) is None
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)

    secondary = object.__new__(SlackAdapter)
    secondary.platform = Platform.SLACK
    runner._profile_adapters = {"wiki": {Platform.SLACK: secondary}}
    source._transport_adapter_ref = weakref.ref(secondary)
    assert run_module._wiki_private_slack_adapter_for_source(runner, source) is None

    class InheritedBaseAdapter(BasePlatformAdapter):
        async def connect(self):
            return None

        async def disconnect(self):
            return None

        async def get_chat_info(self, _chat_id):
            return {}

        async def send(self, *_args, **_kwargs):
            return SendResult(success=True)

    base_adapter = object.__new__(InheritedBaseAdapter)
    base_adapter.platform = Platform.SLACK
    runner.adapters = {Platform.SLACK: base_adapter}
    source._transport_adapter_ref = weakref.ref(base_adapter)
    assert run_module._wiki_private_slack_adapter_for_source(runner, source) is None


def _schedule_on_loop(coro, loop, **_kwargs):
    future = concurrent.futures.Future()

    async def run_coro():
        try:
            future.set_result(await coro)
        except BaseException as exc:  # pragma: no cover - asserted by caller
            future.set_exception(exc)

    loop.call_soon_threadsafe(lambda: asyncio.create_task(run_coro()))
    return future


def _approval_context(source, adapter):
    return SimpleNamespace(
        source=source,
        _status_adapter=adapter,
        _status_chat_id=source.chat_id,
        _status_thread_metadata={"thread_id": source.thread_id},
        _loop_for_step=asyncio.get_running_loop(),
    )


@pytest.mark.asyncio
async def test_wiki_approval_uses_explicit_user_private_notice_and_never_public(
    monkeypatch, tmp_path
):
    _wiki_identity(monkeypatch, tmp_path)
    runner, adapter, source = _live_runner_source()
    monkeypatch.setattr(run_module, "safe_schedule_threadsafe", _schedule_on_loop)
    ctx = _approval_context(source, adapter)

    await asyncio.to_thread(
        run_module._send_wiki_private_approval_sync,
        runner,
        ctx,
        {"allow_permanent": True, "allow_session": True},
        "rm -rf /tmp/example",
        "recursive delete",
    )

    adapter.send_private_notice.assert_awaited_once()
    assert adapter.send_private_notice.await_args.kwargs["user_id"] == "U123"
    adapter.send_exec_approval.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,side_effect",
    [
        (SendResult(success=False, error="private failed"), None),
        (None, RuntimeError("private exception")),
    ],
)
async def test_wiki_private_approval_failure_propagates_without_public_fallback(
    monkeypatch, tmp_path, result, side_effect
):
    _wiki_identity(monkeypatch, tmp_path)
    runner, adapter, source = _live_runner_source()
    if side_effect is not None:
        adapter.send_private_notice = AsyncMock(side_effect=side_effect)
    else:
        adapter.send_private_notice = AsyncMock(return_value=result)
    monkeypatch.setattr(run_module, "safe_schedule_threadsafe", _schedule_on_loop)
    ctx = _approval_context(source, adapter)

    with pytest.raises(RuntimeError):
        await asyncio.to_thread(
            run_module._send_wiki_private_approval_sync,
            runner,
            ctx,
            {},
            "rm -rf /tmp/example",
            "recursive delete",
        )

    adapter.send_exec_approval.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_wiki_operational_private_failure_suppresses_public_send(monkeypatch, tmp_path):
    _wiki_identity(monkeypatch, tmp_path)
    runner, adapter, source = _live_runner_source()
    adapter.send_private_notice = AsyncMock(
        return_value=SendResult(success=False, error="private failed")
    )

    await runner._deliver_platform_notice(source, "operational notice")

    adapter.send_private_notice.assert_awaited_once_with(
        "C123", "U123", "operational notice", metadata={"thread_id": "111.222"}
    )
    adapter.send.assert_not_awaited()
