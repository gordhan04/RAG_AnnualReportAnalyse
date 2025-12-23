#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL=${1:-}
BRANCH=${2:-main}
COMMIT_MESSAGE=${3:-"chore: update"}
FORCE=${4:-}

if ! command -v git >/dev/null 2>&1; then
  echo "git not found" >&2
  exit 1
fi

if [ ! -d .git ]; then
  echo "Initializing git repository"
  git init
fi

if git remote get-url origin >/dev/null 2>&1; then
  EXISTING_URL=$(git remote get-url origin)
  if [ "$EXISTING_URL" != "$REMOTE_URL" ]; then
    git remote remove origin
    git remote add origin "$REMOTE_URL"
  fi
else
  git remote add origin "$REMOTE_URL"
fi

git add .
if git commit -m "$COMMIT_MESSAGE" >/dev/null 2>&1; then
  echo "Committed changes"
else
  echo "No changes to commit"
fi

if [ "$FORCE" = "force" ]; then
  git push -u origin "$BRANCH" --force
else
  git push -u origin "$BRANCH"
fi
