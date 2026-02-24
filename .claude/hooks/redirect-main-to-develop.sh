#!/bin/bash
#
# Claude Code PreToolUse hook: redirect git operations from main → develop
#
# The develop branch is the default integration branch for development.
# This hook intercepts git commands that reference "main" as a branch
# target and rewrites them to use "develop" instead.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Match git commands where "main" is used as a branch reference
# Covers: checkout, switch, pull, merge, rebase with main
# Note: git diff|log are not included since they are helpful to compare main with develop for opening PRs
if echo "$COMMAND" | grep -qE '\bgit\b.*(checkout|switch|pull|merge|rebase)\b.*\bmain\b'; then
  MODIFIED_COMMAND=$(echo "$COMMAND" | perl -pe 's/\bmain\b/develop/g')

  jq -n --arg cmd "$MODIFIED_COMMAND" --arg reason 'Redirected branch reference from "main" to "develop" (develop is the default development branch)' '{
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": "allow",
      "permissionDecisionReason": $reason,
      "updatedInput": {
        "command": $cmd
      }
    }
  }'
  exit 0
fi

# Allow all other commands unchanged
exit 0
