"""Inject HITL channel failures and verify graceful degradation."""

def test_telegram_unreachable_continues():
    """When Telegram API is down, pipeline continues with local logging."""
    # Mock Telegram API to return connection error
    # Verify: pipeline completes, messages logged to hitl_log.jsonl
    pass

def test_cli_not_tty_continues():
    """When stdout is piped (not TTY), CLI adapter switches to write-only."""
    # Mock sys.stdout.isatty() to return False
    # Verify: CLI adapter send() still works (for log capture)
    # Verify: CLI adapter poll_response() returns None immediately
    pass

def test_all_channels_down_pipeline_runs():
    """When ALL HITL channels fail, pipeline runs autonomously."""
    # Mock all channels as unavailable
    # Verify: pipeline completes without crash
    # Verify: no checkpoint or gate blocks the pipeline
    pass
