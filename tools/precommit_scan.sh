#!/bin/sh
# Block commits that carry machine-specific strings into this public
# repository (added 2026-10-11 after a history rewrite for exactly this
# reason: prevention first, rewrite never).
#
# Install with:  cp tools/precommit_scan.sh .git/hooks/pre-commit
# (or symlink; .git/hooks is not versioned, this file is the source).
#
# Install it in every clone of the public tree. The patterns are written
# with a character-class on their last literal so this file never matches
# itself (the same bracket trick used for pattern-safe process kills).

BAD='cjq[0-9]|/data[0-9]/|10\.[0-9]+\.[0-9]+\.[0-9]+|chen_[j]ianqi|yudon[g]|Users.yudon[g]|D:.projec[t]'

staged=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$staged" ] && exit 0

hits=$(git diff --cached -U0 -- $staged | grep -E '^\+' | grep -En "$BAD")
if [ -n "$hits" ]; then
    echo 'pre-commit blocked: machine-specific string(s) in staged changes.'
    echo "$hits" | head -10
    echo 'Fix the source (use arguments/env vars/relative paths), then retry.'
    exit 1
fi
exit 0
