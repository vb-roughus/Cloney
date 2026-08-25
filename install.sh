#!/usr/bin/env bash
# Cloney einrichten (Linux, macOS).
#
# Legt eine virtuelle Umgebung an und übergibt an scripts/setup.py, wo die
# eigentliche Arbeit passiert. Alle Argumente werden durchgereicht, etwa
# --skip-torch oder --dry-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${CLONEY_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  echo "Kein Python gefunden. Cloney braucht Python 3.11 oder neuer." >&2
  exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "$PYTHON ist zu alt. Cloney braucht Python 3.11 oder neuer." >&2
  echo "Anderen Interpreter wählen: CLONEY_PYTHON=python3.12 ./install.sh" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Virtuelle Umgebung anlegen mit $PYTHON ..."
  "$PYTHON" -m venv .venv
fi

exec .venv/bin/python scripts/setup.py "$@"
