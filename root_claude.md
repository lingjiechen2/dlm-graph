# Global Preferences

## Language
- Always respond in Chinese (Simplified) by default.
- Exception: when writing code, code comments, or technical identifiers, use English as normal.
- Switch to another language only if the user explicitly requests it.

## GPU Reservation (sample_gen)
- After finishing any task that used GPUs, immediately re-occupy the freed GPUs using the `hold-gpu` skill (`~/sample_gen.py`) so others cannot claim them.
- Apply to all projects unless the user explicitly says otherwise (e.g. "don't hold GPU", "leave GPU free").
- If unsure which GPUs were just used, ask briefly or default to the GPUs the just-finished job occupied.
- Skip when no GPU was actually used in the task.
