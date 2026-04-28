# Markdown Guidelines

- Use **absolute paths** instead of relative paths.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: source `~/.zshrc` and activate conda env `dllm` (e.g. `conda activate ~/miniconda3/envs/dllm`).
- For tasks requiring a GPU, use the following command: `srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:00 python ...`.

# Communication Guidelines

- If a request is ambiguous or not fully understood, ask concise clarification questions before making changes or running long jobs.
- After any assistant-launched GPU task finishes, immediately start `python /home/lingjie7/sample_gen.py start <gpu_id>` on the same GPU unless the user explicitly says not to.
