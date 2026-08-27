#!/usr/bin/env bash
# Post a new PR comment, or update tensorscope-check's existing one in place,
# identified by the STICKY_MARKER at the top of the body -- never lets
# repeated pushes accumulate a new comment per run.
set -euo pipefail

MARKER="<!-- tensorscope-check -->"

if [[ -z "${PR_NUMBER:-}" ]]; then
  echo "No PR number available for this run -- skipping comment (see the job summary instead)."
  exit 0
fi

if [[ ! -f "${COMMENT_BODY_FILE:?COMMENT_BODY_FILE is required}" ]]; then
  echo "::error::comment body file not found: ${COMMENT_BODY_FILE}"
  exit 1
fi

existing_id="$(
  gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" --paginate \
    --jq "[.[] | select(.body | startswith(\"${MARKER}\"))][0].id // empty"
)"

if [[ -n "${existing_id}" ]]; then
  echo "Updating existing sticky comment ${existing_id}"
  gh api "repos/${GITHUB_REPOSITORY}/issues/comments/${existing_id}" \
    -X PATCH -F body=@"${COMMENT_BODY_FILE}" > /dev/null
else
  echo "Posting new sticky comment"
  gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
    -X POST -F body=@"${COMMENT_BODY_FILE}" > /dev/null
fi
