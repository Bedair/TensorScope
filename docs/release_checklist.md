# Release checklist

1. Run `make -C tools/tflm_oracle`, `pytest -q`, and `git diff --check`.
2. Build with `python -m build` and inspect wheel/sdist contents.
3. Install the wheel in a clean temporary environment and smoke-test
   `tensorscope --version` and `tensorscope analyze`.
4. Confirm generated oracle binaries and temporary reports are excluded from
   the release commit.
5. Review schemas, exit codes, pinned revision, operator coverage, and all
   limitations. The user performs tagging and publishing separately.
