# tools/operator_channel.py

import sys
import json
import time
import logging
import threading
import queue
import requests
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Channel Adapter Base ─────────────────────────────────────────────────────

class ChannelAdapter(ABC):
    @abstractmethod
    def send(self, message: str, level: str, data: dict = None) -> None:
        """Send a message to the operator."""
        pass

    @abstractmethod
    def poll_response(self, timeout: int) -> Optional[str]:
        """Wait for operator response, return None on timeout."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if channel is functional."""
        pass

# ── CLI Adapter ──────────────────────────────────────────────────────────────

class CLIAdapter(ChannelAdapter):
    def __init__(self):
        self.response_queue = queue.Queue()
        self.is_tty = sys.stdout.isatty()
        
        # Colors (ANSI)
        self.colors = {
            "STATUS": "\033[36m",      # Cyan
            "CHECKPOINT": "\033[33m",  # Yellow
            "GATE": "\033[31m",        # Red (for gate)
            "ESCALATION": "\033[1;31m",# Bold Red
            "RESULT": "\033[32m",      # Green
            "RESET": "\033[0m",
            "BOLD": "\033[1m"
        }

    def send(self, message: str, level: str, data: dict = None) -> None:
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        color = self.colors.get(level, self.colors["RESET"])
        
        # Format scores/results specifically
        if level == "RESULT":
            formatted_msg = f"{timestamp} {color}{self.colors['BOLD']}{message}{self.colors['RESET']}"
        else:
            formatted_msg = f"{timestamp} {color}{message}{self.colors['RESET']}"
            
        print(formatted_msg)
        sys.stdout.flush()

    def poll_response(self, timeout: int) -> Optional[str]:
        if not self.is_tty:
            return None
            
        # Start a background thread for non-blocking input
        # Note: In a real CLI, we might use something like `select` or `prompt_toolkit`,
        # but for a standard script, a thread + input() is common.
        def get_input():
            try:
                res = input(">> ")
                self.response_queue.put(res)
            except EOFError:
                pass

        thread = threading.Thread(target=get_input, daemon=True)
        thread.start()
        
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_available(self) -> bool:
        return True # Always print to logs even if not TTY

# ── Telegram Adapter ─────────────────────────────────────────────────────────

class TelegramAdapter(ChannelAdapter):
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.update_offset = 0
        self.local_mode = False
        self._last_check_time = 0

    def send(self, message: str, level: str, data: dict = None) -> None:
        if self.local_mode:
            self._log_to_local(message, level, data)
            return

        try:
            # Handle photo if data contains an image path
            if data and "image_path" in data:
                with open(data["image_path"], "rb") as f:
                    resp = requests.post(
                        f"{self.base_url}/sendPhoto",
                        params={"chat_id": self.chat_id, "caption": message},
                        files={"photo": f},
                        timeout=10
                    )
            else:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": message},
                    timeout=10
                )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}. Switching to LOCAL mode.")
            self.local_mode = True
            self._last_check_time = time.time()
            self._log_to_local(message, level, data)

    def poll_response(self, timeout: int) -> Optional[str]:
        if self.local_mode:
            self._check_reconnect()
            return None

        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={"offset": self.update_offset, "timeout": min(timeout, 50)},
                timeout=timeout + 5
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
            
            for update in updates:
                self.update_offset = update["update_id"] + 1
                if "message" in update and str(update["message"]["chat"]["id"]) == str(self.chat_id):
                    return update["message"].get("text")
        except Exception as e:
            logger.warning(f"Telegram poll failed: {e}")
            
        return None

    def _log_to_local(self, message: str, level: str, data: dict):
        with open("hitl_log.jsonl", "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "data": data
            }) + "\n")

    def _check_reconnect(self):
        if time.time() - self._last_check_time > 60:
            if self.is_available():
                self.local_mode = False
                self.send("🔄 Telegram connection restored. Local mode deactivated.", "STATUS")

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/getMe", timeout=5)
            return resp.status_code == 200
        except:
            return False

# ── Command Listener & Manager ───────────────────────────────────────────────

class CommandListener:
    def __init__(self, manager: "ChannelManager"):
        self.manager = manager
        self.stop_event = threading.Event()
        self.thread = None
        
        # Shared flags for background commands
        self.pipeline_paused = False
        self.pipeline_aborted = False
        self.injection_queue = queue.Queue()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        while not self.stop_event.is_set():
            # Long poll across all channels
            # Note: For simplicity, we poll them in sequence with a short timeout
            response = self.manager.poll_any(timeout=2)
            if response:
                self._classify_command(response)
            time.sleep(0.5)

    def _classify_command(self, text: str):
        if text.startswith("/pause"):
            self.pipeline_paused = True
            self.manager.send_all("⏸️ Pipeline paused by operator.", "STATUS")
        elif text.startswith("/resume"):
            self.pipeline_paused = False
            self.manager.send_all("▶️ Pipeline resumed.", "STATUS")
        elif text.startswith("/abort"):
            self.pipeline_aborted = True
            self.manager.send_all("🛑 Pipeline abortion requested.", "STATUS")
        elif text.startswith("/status"):
            # This will be handled by the next transition hook reading state
            self.injection_queue.put({"type": "command", "cmd": "status"})
        elif text.startswith("/feature "):
            hint = text[len("/feature "):].strip()
            self.injection_queue.put({"type": "feature_hint", "text": hint})
            self.manager.send_all(f"📝 Feature hint recorded: {hint}", "STATUS")
        elif text.startswith("/override "):
            # Format: /override field_name value
            parts = text[len("/override "):].strip().split(" ", 1)
            if len(parts) == 2:
                self.injection_queue.put({"type": "override", "field": parts[0], "value": parts[1]})
                self.manager.send_all(f"⚡ Override queued: {parts[0]} = {parts[1]}", "STATUS")
        elif text.startswith("/skip "):
            agent = text[len("/skip "):].strip()
            self.injection_queue.put({"type": "skip", "agent": agent})
            self.manager.send_all(f"⏭️ Skip queued for: {agent}", "STATUS")
        elif not text.startswith("/"):
            # Domain knowledge injection
            self.injection_queue.put({"type": "domain", "text": text})
            self.manager.send_all(f"🧠 Domain knowledge recorded: {text}", "STATUS")

class ChannelManager:
    def __init__(self, channels: List[str], config: dict):
        self.adapters: List[ChannelAdapter] = []
        for ch in channels:
            if ch == "cli":
                self.adapters.append(CLIAdapter())
            elif ch == "telegram":
                bot_token = config.get("PROFESSOR_TELEGRAM_BOT_TOKEN")
                chat_id = config.get("PROFESSOR_TELEGRAM_CHAT_ID")
                if bot_token and chat_id:
                    self.adapters.append(TelegramAdapter(bot_token, chat_id))
        
        self.listener = CommandListener(self)
        self.msg_sequence = 0

    def send_all(self, message: str, level: str, data: dict = None) -> None:
        self.msg_sequence += 1
        formatted_msg = f"#{self.msg_sequence} {message}"
        if len(formatted_msg) > 500:
            formatted_msg = formatted_msg[:497] + "..."
            
        for adapter in self.adapters:
            adapter.send(formatted_msg, level, data)

    def poll_any(self, timeout: int) -> Optional[str]:
        # Simple implementation: poll each and return first
        # In multi-channel, we could use threads to poll simultaneously
        for adapter in self.adapters:
            res = adapter.poll_response(timeout=1) # short poll
            if res:
                return res
        return None

    def start_listener(self):
        self.listener.start()

    def stop_listener(self):
        self.listener.stop()

# ── Global Singleton & API ───────────────────────────────────────────────────

_MANAGER: Optional[ChannelManager] = None

def init_hitl(channels: List[str], config: dict) -> ChannelManager:
    global _MANAGER
    _MANAGER = ChannelManager(channels, config)
    _MANAGER.start_listener()
    return _MANAGER

def emit_to_operator(message: str, level: str = "STATUS", data: dict = None) -> Optional[str]:
    if not _MANAGER:
        return None
        
    _MANAGER.send_all(message, level, data)
    
    # Behavior based on level
    if level in ["STATUS", "RESULT"]:
        return None
        
    if level == "CHECKPOINT":
        return _MANAGER.poll_any(timeout=180)
        
    if level == "GATE":
        return _MANAGER.poll_any(timeout=900)
        
    if level == "ESCALATION":
        # Block indefinitely until response
        while True:
            res = _MANAGER.poll_any(timeout=10)
            if res:
                return res
            time.sleep(1)

def process_pending_injections(state: Any) -> Any:
    """Checks the injection queue and applies items to state."""
    if not _MANAGER:
        return state
        
    updated_fields = {}
    
    while not _MANAGER.listener.injection_queue.empty():
        item = _MANAGER.listener.injection_queue.get()
        t_str = datetime.now().isoformat()
        
        if item["type"] == "domain":
            injections = state.get("hitl_injections", [])
            injections.append({"text": item["text"], "timestamp": t_str, "injected_into_agents": []})
            updated_fields["hitl_injections"] = injections
            
        elif item["type"] == "feature_hint":
            hints = state.get("hitl_feature_hints", [])
            hints.append({"text": item["text"], "status": "pending", "rejection_reason": ""})
            updated_fields["hitl_feature_hints"] = hints
            
        elif item["type"] == "override":
            overrides = state.get("hitl_overrides", {})
            overrides[item["field"]] = item["value"]
            updated_fields["hitl_overrides"] = overrides
            
        elif item["type"] == "skip":
            skips = state.get("hitl_skip_agents", [])
            skips.append(item["agent"])
            updated_fields["hitl_skip_agents"] = skips

    # Apply flags
    updated_fields["pipeline_paused"] = _MANAGER.listener.pipeline_paused
    updated_fields["pipeline_aborted"] = _MANAGER.listener.pipeline_aborted

    if updated_fields:
        # Import state here to avoid circular dependency
        from core.state import ProfessorState
        return ProfessorState.validated_update(state, "hitl_listener", updated_fields)
        
    return state
