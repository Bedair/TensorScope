# Release checklist

1. Run `make -C tools/tflm_oracle`, `pytest -q`, and `git diff --check`.
2. Build with `python -m build`, run `python -m twine check dist/*`, and inspect
   wheel/sdist contents.
3. Install the wheel in a clean temporary environment and smoke-test
   `tensorscope --version`, `analyze`, `validate`, and `compare`.
4. Confirm generated oracle binaries and temporary reports are excluded from
   the release commit.
5. Review schemas, exit codes, pinned revision, operator coverage, and all
   limitations. The user performs tagging and publishing separately.
6. Recheck wheel reproducibility with a fixed `SOURCE_DATE_EPOCH`; record sdist
   reproducibility separately rather than treating it as equivalent.
