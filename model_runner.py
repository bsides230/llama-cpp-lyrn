import os
import sys
import time
import datetime
import json
import signal
import threading
import contextlib
import io
import re
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llama_cpp import Llama
from llama_cpp.llama_cache import LlamaRAMCache

try:
    from .settings_manager import SettingsManager
    from .snapshot_loader import SnapshotLoader
    from .delta_manager import DeltaManager
    from .chat_manager import ChatManager
    from .automation_controller import AutomationController
except ImportError:
    from settings_manager import SettingsManager
    from snapshot_loader import SnapshotLoader
    from delta_manager import DeltaManager
    from chat_manager import ChatManager
    from automation_controller import AutomationController

# Global flag for clean shutdown
running = True
model_lock = threading.Lock()

def signal_handler(sig, frame):
    global running
    print("Shutting down worker...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_FILE = os.path.join(SCRIPT_DIR, "data", "chat_trigger.txt")
STOP_TRIGGER = os.path.join(SCRIPT_DIR, "data", "stop_trigger.txt")
REBUILD_TRIGGER = os.path.join(SCRIPT_DIR, "data", "rebuild_trigger.txt")
LLM_STATUS_FILE = os.path.join(SCRIPT_DIR, "data", "global_flags", "llm_status.txt")
STATS_FILE = os.path.join(SCRIPT_DIR, "data", "global_flags", "llm_stats.json")
LAST_ERROR_FILE = os.path.join(SCRIPT_DIR, "data", "global_flags", "last_error.txt")

def set_llm_status(status: str):
    try:
        os.makedirs(os.path.dirname(LLM_STATUS_FILE), exist_ok=True)
        with open(LLM_STATUS_FILE, 'w', encoding='utf-8') as f:
            f.write(status)
    except Exception as e:
        print(f"Error setting LLM status: {e}")

def write_stats(stats_data):
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f)
    except Exception as e:
        print(f"Error writing stats: {e}")

def write_error(error_msg: str):
    try:
        with open(LAST_ERROR_FILE, 'w', encoding='utf-8') as f:
            f.write(error_msg)
    except Exception as e:
        print(f"Error writing last error: {e}")

def parse_metrics(log_output: str):
    stats = {}
    try:
        # KV Cache
        kv_match = re.search(r'(\d+)\s+prefix-match hit', log_output)
        if kv_match:
            stats["kv_cache_reused"] = int(kv_match.group(1))

        # Prompt Eval
        prompt_match = re.search(r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*ms per token', log_output)
        if prompt_match:
            ms = float(prompt_match.group(1))
            tokens = int(prompt_match.group(2))
            ms_per_tok = float(prompt_match.group(3))
            stats["tokenization_time_ms"] = ms
            stats["prompt_tokens"] = tokens
            stats["prompt_speed"] = 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0

        # Eval (Generation)
        eval_match = re.search(r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*ms per token', log_output)
        if eval_match:
            ms = float(eval_match.group(1))
            tokens = int(eval_match.group(2))
            ms_per_tok = float(eval_match.group(3))
            stats["generation_time_ms"] = ms
            stats["eval_tokens"] = tokens
            stats["eval_speed"] = 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0

        # Load Time
        load_match = re.search(r'load time\s*=\s*([\d.]+)\s*ms', log_output)
        if load_match:
            stats["load_time"] = float(load_match.group(1))

        # Total Time
        total_match = re.search(r'total time\s*=\s*([\d.]+)\s*ms', log_output)
        if total_match:
            stats["total_time"] = float(total_match.group(1)) / 1000.0

        if "prompt_tokens" in stats and "eval_tokens" in stats:
            stats["total_tokens"] = stats["prompt_tokens"] + stats["eval_tokens"]

    except Exception:
        pass
    return stats

def attempt_load_model(settings_manager):
    """Attempts to load the model based on current settings."""
    settings = settings_manager.settings
    active_config = settings.get("active", {})
    model_path_setting = active_config.get("model_path", "").strip()

    if not model_path_setting:
         print("Warning: No model path configured.")
         set_llm_status("error")
         write_error("No model path configured.")
         return None, None

    resolved_path = os.path.join(SCRIPT_DIR, "models", model_path_setting)

    if not os.path.exists(resolved_path):
        print(f"Error: Model not found: {resolved_path}")
        set_llm_status("error")
        write_error(f"Model not found: {resolved_path}")
        return None, model_path_setting

    print(f"Loading model: {resolved_path}")
    set_llm_status("loading")

    try:
        llm = Llama(
            model_path=resolved_path,
            n_ctx=active_config.get("n_ctx", 2048),
            n_threads=active_config.get("n_threads", 4),
            n_gpu_layers=active_config.get("n_gpu_layers", 0),
            n_batch=active_config.get("n_batch", 512),
            use_mlock=True,
            use_mmap=False,
            chat_format=active_config.get("chat_format"),
            add_bos=True,
            add_eos=True,
            verbose=True
        )
        # Enable RAM-based KV cache to preserve state across completions.
        # This is critical for models (like Qwen) whose memory backend does
        # not support partial KV cache removal, as it allows state
        # save/restore to bypass that limitation.
        cache_size = active_config.get("cache_size_bytes", 2 << 30)  # default 2 GB
        llm.set_cache(LlamaRAMCache(capacity_bytes=cache_size))
        print("Model loaded successfully.")
        set_llm_status("idle")
        return llm, model_path_setting
    except Exception as e:
        print(f"Failed to load model: {e}")
        set_llm_status("error")
        write_error(f"Failed to load model: {e}")
        return None, model_path_setting

def main():
    print("--- Model Runner Starting (v5 Logic) ---")

    # 1. Initialize Managers
    settings_manager = SettingsManager()

    # Reload settings
    settings_manager.load_or_detect_first_boot()
    settings = settings_manager.settings

    automation_controller = AutomationController()
    snapshot_loader = SnapshotLoader(settings_manager, automation_controller)
    delta_manager = DeltaManager()

    role_mappings = {
        "assistant": "final_output",
        "model": "final_output",
        "thinking": "thinking_process",
        "analysis": "thinking_process"
    }

    chat_dir = settings.get("paths", {}).get("chat", "chat")
    chat_manager = ChatManager(chat_dir, settings_manager, role_mappings)

    # 2. Initial Model Load Attempt
    # We do NOT return if this fails. We just loop.
    llm, loaded_model_setting = attempt_load_model(settings_manager)

    # 3. Main Loop
    print(f"Watching for trigger: {TRIGGER_FILE}")

    while running:
        # Check for Rebuild Trigger
        if os.path.exists(REBUILD_TRIGGER):
            print(f"[Runner] Rebuild trigger detected.")
            try:
                os.remove(REBUILD_TRIGGER)
                settings_manager.load_or_detect_first_boot()
                settings = settings_manager.settings
                snapshot_loader.build_master_prompt_from_components()
                print("[Runner] Snapshot rebuilt.")

                # Check if model config changed or we have no model
                current_model_setting = settings.get("active", {}).get("model_path", "")
                if llm is None or current_model_setting != loaded_model_setting:
                    print("[Runner] Configuration changed or no model loaded. Attempting reload.")
                    llm, loaded_model_setting = attempt_load_model(settings_manager)

            except Exception as e:
                print(f"[Runner] Error rebuilding/reloading: {e}")

        if os.path.exists(TRIGGER_FILE):
            print(f"[Runner] Trigger detected: {TRIGGER_FILE}")

            # Reload settings
            try:
                settings_manager.load_or_detect_first_boot()
                settings = settings_manager.settings
            except Exception:
                pass

            # Check if we have a model. If not, try loading one last time
            if llm is None:
                 print("[Runner] No model loaded. Attempting late load...")
                 llm, loaded_model_setting = attempt_load_model(settings_manager)

            try:
                with open(TRIGGER_FILE, 'r', encoding='utf-8') as f:
                    chat_file_path_str = f.read().strip()

                try:
                    os.remove(TRIGGER_FILE)
                except OSError: pass

                if chat_file_path_str:
                    if llm:
                        process_request(llm, chat_file_path_str, snapshot_loader, delta_manager, chat_manager, settings_manager)
                    else:
                        print("[Runner] Cannot process request: No model loaded.")
                        # Write helpful error to chat
                        try:
                            with open(chat_file_path_str, "a", encoding="utf-8") as f:
                                f.write("\n\n[Error: No model loaded. Please configure a valid model path in Settings.]\n")
                        except: pass
                        set_llm_status("error")

            except Exception as e:
                print(f"Error processing trigger: {e}")
                set_llm_status("error")

        time.sleep(0.1)

    print("Runner stopped.")
    set_llm_status("stopped")

def process_request(llm, chat_file_path_str: str, snapshot_loader, delta_manager, chat_manager, settings_manager):
    with model_lock:
        set_llm_status("busy")
        print(f"[Runner DEBUG {datetime.datetime.now()}] Processing request from: {chat_file_path_str}")

        # Cleanup stop trigger
        try:
            if os.path.exists(STOP_TRIGGER):
                os.remove(STOP_TRIGGER)
        except: pass

        chat_file_path = Path(chat_file_path_str)
        if not chat_file_path.exists():
            print(f"[Runner ERROR {datetime.datetime.now()}] Chat file not found: {chat_file_path}")
            set_llm_status("idle")
            return

        try:
            # 1. Read User Message from the Triggered File
            # start_lyrn.py writes: user\n{message}\n
            try:
                content = chat_file_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"[Runner ERROR {datetime.datetime.now()}] Failed to read chat file: {e}")
                set_llm_status("error")
                return

            user_message = ""
            # Check for v4 format first
            if content.startswith("user\n"):
                user_message = content[5:].strip() # Remove 'user\n'
            else:
                # Fallback to old regex if needed or raw
                match = re.search(r"#USER_START#\n(.*?)\n#USER_END#", content, re.DOTALL)
                if match:
                    user_message = match.group(1).strip()
                else:
                    user_message = content.strip()

            print(f"[Runner DEBUG {datetime.datetime.now()}] User Message extracted: {user_message[:50]}...")

            # 2. Build Context (v4 Logic)
            # Master Prompt
            system_prompt = snapshot_loader.load_base_prompt()

            # Deltas
            delta_content = ""
            if settings_manager.get_setting("enable_deltas", True):
                delta_content = delta_manager.get_delta_content()

            # Construct Messages
            # Order: Snapshot -> History -> Deltas -> New Input
            messages = [{"role": "system", "content": system_prompt}]

            # History
            # IMPORTANT: Exclude the current chat file so we don't duplicate it or treat it as history yet
            history = chat_manager.get_chat_history_messages(exclude_paths=[str(chat_file_path.resolve())])
            messages.extend(history)

            # Deltas (Injected after history, before new input)
            if delta_content:
                messages.append({"role": "system", "content": delta_content})

            # Append current user message
            # Merge if last was user (alternating roles logic)
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] += "\n\n" + user_message
            else:
                messages.append({"role": "user", "content": user_message})

            # 3. Generate with Stderr Capture
            active_config = settings_manager.settings.get("active", {})
            log_capture_buffer = io.StringIO()

            print(f"[Runner DEBUG {datetime.datetime.now()}] Generating response... (Use mlock: True, mmap: False)")

            with contextlib.redirect_stderr(log_capture_buffer):
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=active_config.get("max_tokens", 2048),
                    temperature=active_config.get("temperature", 0.7),
                    top_p=active_config.get("top_p", 0.95),
                    top_k=active_config.get("top_k", 40),
                    stream=True
                )

                # 4. Stream Response to File
                # Prepend model separator
                with open(chat_file_path, "a", encoding="utf-8") as f:
                    f.write("\n\nmodel\n")

                    token_count = 0
                    for token_data in stream:
                        # Check stop
                        if os.path.exists(STOP_TRIGGER):
                            print(f"[Runner DEBUG {datetime.datetime.now()}] Stop trigger detected.")
                            try:
                                os.remove(STOP_TRIGGER)
                            except: pass
                            f.write("\n\n[Stopped]")
                            break

                        if 'choices' in token_data and len(token_data['choices']) > 0:
                            delta = token_data['choices'][0].get('delta', {})
                            text = delta.get('content', '')
                            if text:
                                f.write(text)
                                f.flush()
                                token_count += 1

            # 5. Parse Metrics from Captured Log
            log_output = log_capture_buffer.getvalue()
            # Print to real stderr so start_lyrn can capture it too
            print(log_output, file=sys.stderr)

            stats = parse_metrics(log_output)
            if stats:
                write_stats(stats)

            print("Generation complete.")
            set_llm_status("idle")

        except Exception as e:
            print(f"[Runner ERROR {datetime.datetime.now()}] Error during generation: {e}")
            import traceback
            traceback.print_exc()
            try:
                with open(chat_file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[Error: {e}]\n")
            except: pass
            set_llm_status("error")

if __name__ == "__main__":
    main()
