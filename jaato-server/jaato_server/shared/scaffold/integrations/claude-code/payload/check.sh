#!/usr/bin/env bash
# Every `jaato-scaffold explain <topic>` this skill names must resolve against
# the INSTALLED framework.  A skill that cites a topic the framework dropped is
# worse than no skill: it sends the reader to a dead end and they reach for the
# source, which is the exact failure this skill exists to prevent.
#
# Run from the same environment as the daemon.  Non-zero exit on any miss.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

topics=$(grep -ohE 'jaato-scaffold explain [a-z-]+( <?[a-z_-]+>?)?' "$here"/SKILL.md "$here"/references/*.md \
         | sed 's/jaato-scaffold explain //' | sed 's/ <.*>//' | sort -u)

while read -r t; do
  [ -z "$t" ] && continue
  # `plugin`/`provider`/`archetype` take an argument; the bare form is a usage
  # stub, so checking it proves nothing.  The concrete forms are checked below.
  case "$t" in plugin|provider|archetype) continue;; esac
  out=$(jaato-scaffold explain $t 2>&1 | grep -vE '^\[(info|warn|error)\]')
  n=$(printf '%s' "$out" | grep -c . || true)
  if printf '%s' "$out" | grep -qiE "^unknown |unknown (topic|plugin|provider|archetype)"; then
    echo "MISS  explain $t — unknown topic"; fail=1
  elif [ "$n" -lt 2 ]; then
    # Empty output is the failure mode a naive check calls success.
    echo "MISS  explain $t — resolved but said nothing ($n lines)"; fail=1
  else
    echo "ok    explain $t  ($n lines)"
  fi
done <<< "$topics"

for a in profile-set cascade client fire host-tools observer processor sweep; do
  if jaato-scaffold explain archetype "$a" 2>&1 | grep -qi '^unknown'; then
    echo "MISS  archetype $a"; fail=1
  else
    echo "ok    archetype $a"
  fi
done

[ $fail -eq 0 ] && echo "PASS — every cited topic resolves" || echo "FAIL — the skill cites topics the framework does not have"
exit $fail
