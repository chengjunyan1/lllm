# Logging & Observability

Agents are only as good as their observability. LLLM includes a replayable logging stack so every message, function call, and frontend event can be audited after the fact.

## Collections & Sessions

`lllm.const.RCollections` defines three canonical collections:

- `dialogs` – one entry per dialog id created during a session.
- `messages` – the full transcript for each dialog (`session/dialog_id`).
- `frontend` – arbitrary UI or status events (used by `StreamWrapper`).

A `Dialog` automatically logs every appended `Message` along with metadata (creator, role, cost, parsed payload). Forked dialogs inherit the log base so their lineage is tracked.

## Log Base Implementations

`lllm/log.py` provides a pluggable abstraction:

- `ReplayableLogBase` (abstract) – enforces collection names and powers `ReplaySession.activities` for chronological replays.
- `LocalFileLog` – stores JSON blobs on disk under `<log_dir>/<collection>/<session>/<timestamp>.json`.
- `NoLog` – disables persistence while keeping the API surface identical.

`build_log_base(config)` chooses an implementation based on the YAML configuration (`log_type: localfile | none`). Because the log base is passed to every agent, switching persistence strategies requires no code changes.

## Stream & Frontend Helpers

`lllm.utils.PrintSystem` and `StreamWrapper` mirror a minimal Streamlit-like interface:

- `.write`, `.markdown`, `.spinner`, `.expander`, `.divider`, `.code` etc. log frontend events into the `frontend` collection.
- Agents call `self.set_st(session_name)` before each task to associate logs with a human-readable session id.

When building custom UIs, implement the same methods and forward them to your logging or telemetry stack.

## Replay Workflows

`ReplaySession` stitches dialogs and frontend entries into a time-sorted list of `Activity` objects. This enables:

- Rich transcripts (who said what, which prompt triggered the response, which tool ran).
- Time-based debugging (spot retries, rate-limit pauses, etc.).
- Lightweight analytics (count function calls per agent, measure exception rates).

Because every `Message` stores `usage` and `CompletionCost`, you can compute spend per run without instrumenting each API call manually.

## Configuration Tips

- Keep `log_dir` outside version control (default `./logs`).
- Set `log_type: none` for unit tests to avoid filesystem noise.
- For long-running systems, use a persistent volume and prune old sessions by deleting directories under `logs/`.
- Combine logging with sandbox metadata: `JupyterSession.to_dict()` captures proxy activation and cutoff dates so a replay can rebuild the exact environment.
