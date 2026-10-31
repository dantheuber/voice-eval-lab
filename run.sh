#!/usr/bin/env bash
# Starts voice-eval-lab and stops everything it started when you press Ctrl-C.
#
#   ./run.sh                     local-default, all model servers loaded before you start
#   ./run.sh --profile NAME      another profile from config/profiles
#   ./run.sh --lazy              start model servers only when a call needs them
#   ./run.sh --fake              no models, for a quick look. Data stays apart, in data/fake
#   ./run.sh --http-port N       dashboard port on this machine, default 18080
#   ./run.sh --https-port N      LAN port, default 18443
#   ./run.sh --check             check prerequisites and ports, then stop without starting

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

STOP_GRACE_S=45 # one model server gets 10 s, then the dashboard's own connections have to close
HEALTH_LIMIT_S=120
MODEL_LIMIT_S=600

PROFILE=local-default
FAKE=0
LAZY=0
CHECK_ONLY=0
HTTP_PORT=18080
HTTPS_PORT=18443

say() { printf '%s\n' "$*"; }
die() {
  say "$*" >&2
  exit 1
}
usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      [ $# -ge 2 ] || die "--profile needs a value"
      PROFILE="$2"
      shift 2
      ;;
    --http-port | --https-port)
      [ $# -ge 2 ] || die "$1 needs a value"
      [ "$1" = --http-port ] && HTTP_PORT="$2" || HTTPS_PORT="$2"
      shift 2
      ;;
    --fake) FAKE=1 && shift ;;
    --lazy) LAZY=1 && shift ;;
    --check) CHECK_ONLY=1 && shift ;;
    -h | --help) usage && exit 0 ;;
    *)
      usage >&2
      die "Unknown option: $1"
      ;;
  esac
done

if [ "$FAKE" = 1 ]; then
  PROFILE=fake
  # Fake calls must not land in real pass rates, and fake must not become the saved profile.
  export VOICELAB_DATA_DIR="${VOICELAB_DATA_DIR:-$ROOT/data/fake}"
fi
DATA_DIR="${VOICELAB_DATA_DIR:-$ROOT/data}"
LOG="$DATA_DIR/logs/serve.log"
BASE="http://127.0.0.1:$HTTP_PORT"

PROFILE_FILE="$(grep -lE "^id: *\"?$PROFILE\"?\s*$" config/profiles/*.yaml 2> /dev/null | head -n1)"
if [ -z "$PROFILE_FILE" ]; then
  say "No profile named '$PROFILE'. These exist:" >&2
  sed -n 's/^id: *//p' config/profiles/*.yaml | sed 's/^/  /' >&2
  exit 1
fi
# "port name" for each model server. The profile files keep these keys at fixed indents.
SERVICES="$(awk '/^  - name:/ {name = $3} /^    port:/ {print $2, name}' "$PROFILE_FILE")"

port_owner() { # prints who holds a port, and nothing if it is free
  local listing
  listing="$(ss -H -ltnp "sport = :$1" 2> /dev/null | head -n1)"
  [ -n "$listing" ] || return 0
  if [[ "$listing" =~ users:\(\(\"([^\"]+)\",pid=([0-9]+) ]]; then
    local where=""
    [ "$(readlink "/proc/${BASH_REMATCH[2]}/cwd" 2> /dev/null)" = "$ROOT" ] && where=", from this checkout"
    say "${BASH_REMATCH[1]} (pid ${BASH_REMATCH[2]}$where)"
  else
    say "a process owned by another user"
  fi
}

web_is_stale() {
  [ -f web/dist/index.html ] || return 0
  [ -n "$(find web/src web/index.html web/package.json web/pnpm-lock.yaml web/vite.config.ts \
    web/tsconfig.json web/tsconfig.app.json web/tsconfig.node.json \
    -newer web/dist/index.html -print -quit 2> /dev/null)" ]
}

check_prerequisites() {
  local tool
  for tool in curl ss; do
    command -v "$tool" > /dev/null || die "$tool is missing. Install it with your package manager."
  done
  command -v uv > /dev/null ||
    die "uv is missing. It installs without sudo: curl -LsSf https://astral.sh/uv/install.sh | sh"

  if web_is_stale && { ! command -v node > /dev/null || ! command -v pnpm > /dev/null; }; then
    if [ -f web/dist/index.html ]; then
      say "Note: web/dist is older than web/src, and node or pnpm is missing, so it stays as it is."
    else
      die "The dashboard isn't built, and that needs node and pnpm. Install them, then run this again."
    fi
  fi

  [ -n "$SERVICES" ] || return 0
  case "$PROFILE" in
    local-default | local-fast | local-realtime) ;;
    *) return 0 ;;
  esac
  local missing
  if ! missing="$(scripts/setup.sh --check --quiet --profile "$PROFILE")"; then
    say "Profile $PROFILE is not set up. Missing:" >&2
    say "$missing" >&2
    die "Run: scripts/setup.sh --profile $PROFILE"
  fi
  if [ -n "$missing" ]; then
    say "Note: some optional files are not here yet. They download on first use, or get them now"
    say "with scripts/setup.sh --profile $PROFILE:"
    say "$missing"
  fi
}

check_ports() {
  local -a taken=()
  local port name owner
  while read -r port name; do
    [ -n "$port" ] || continue
    owner="$(port_owner "$port")"
    [ -z "$owner" ] || taken+=("$(printf '  %-6s %-14s %s' "$port" "$name" "$owner")")
  done <<< "$HTTP_PORT dashboard
$HTTPS_PORT dashboard-lan
$SERVICES"
  [ "${#taken[@]}" -gt 0 ] || return 0
  say "Not starting, because these ports are already in use:" >&2
  printf '%s\n' "${taken[@]}" >&2
  say "If that is another ./run.sh, stop it with Ctrl-C in its terminal." >&2
  say "Dashboard ports can move: ./run.sh --http-port 28080 --https-port 28443" >&2
  say "Model server ports are set in $PROFILE_FILE." >&2
  exit 1
}

check_prerequisites
check_ports
if [ "$CHECK_ONLY" = 1 ]; then
  say "Prerequisites look fine for profile $PROFILE, and the ports are free."
  web_is_stale && say "web/dist would be rebuilt."
  exit 0
fi

# --inexact keeps optional extras, such as the aws one from docs/ADAPTERS.md.
say "Syncing Python packages"
uv sync --inexact --quiet || die "uv sync failed."

if web_is_stale && command -v pnpm > /dev/null; then
  say "Building the dashboard"
  (cd web && pnpm install --frozen-lockfile && pnpm build) || die "The dashboard build failed."
fi

mkdir -p "$(dirname "$LOG")"
[ ! -f "$LOG" ] || mv -f "$LOG" "$LOG.1"

SERVER_PID=""
STOP_ASKED=0
STOP_DEADLINE=0

force_stop() {
  pkill -KILL -P "$SERVER_PID" 2> /dev/null
  kill -KILL "$SERVER_PID" 2> /dev/null
}

on_signal() {
  STOP_ASKED=$((STOP_ASKED + 1))
  [ -n "$SERVER_PID" ] || exit 130
  note_children
  if [ "$STOP_ASKED" -eq 1 ]; then
    say ""
    say "Stopping. Model servers shut down first. Press Ctrl-C again to force it."
    STOP_DEADLINE=$((SECONDS + STOP_GRACE_S))
    kill -TERM "$SERVER_PID" 2> /dev/null
  else
    force_stop
  fi
}
trap on_signal INT TERM HUP

arguments=(serve --profile "$PROFILE" --http-port "$HTTP_PORT" --https-port "$HTTPS_PORT")
[ "$LAZY" = 0 ] || arguments+=(--services lazy)

say "Starting profile $PROFILE. Server log: ${LOG#"$ROOT"/}"
uv run --no-sync voicelab "${arguments[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

print_urls() {
  local system lan
  system="$(curl -fs -m 3 --noproxy '*' "$BASE/api/system")"
  lan="$(sed -n 's/.*"lan":"\([^"]*\)".*/\1/p' <<< "$system")"
  say ""
  say "voice-eval-lab is up, profile $PROFILE."
  say "  This machine:  $BASE"
  say "  LAN:           ${lan:-https://this-machine:$HTTPS_PORT}"
  say "    The LAN address uses a self-signed certificate. The browser warns once, and you have"
  say "    to accept it, because browsers only give a page the mic on localhost or HTTPS."
  say "  Press Ctrl-C to stop. Model servers stop with it."
  say ""
}

report_models() { # prints progress when it changes, and returns 0 once every role is ready
  local health total ready down
  health="$(curl -fs -m 3 --noproxy '*' "$BASE/api/health")" || return 1
  [[ "$health" == *'"ready":true'* ]] && return 0
  total="$(grep -o '"state":"' <<< "$health" | wc -l)"
  ready="$(grep -o '"state":"ok"' <<< "$health" | wc -l)"
  down="$(grep -o '"name":"[^"]*","model":"[^"]*","state":"down","detail":"[^"]*"' <<< "$health" |
    sed 's/"name":"\([^"]*\)".*"detail":"\([^"]*\)"/  \1: \2/')"
  local progress="Loading models: $ready of $total roles ready."
  [ -z "$down" ] || progress+=$'\n'"$down"$'\n'"  Logs are in ${DATA_DIR#"$ROOT"/}/logs/"
  if [ "$progress" != "$LAST_PROGRESS" ]; then
    say "$progress"
    LAST_PROGRESS="$progress"
  fi
  return 1
}

# Everything voicelab started, by pid, with the command line it had. A model server runs in its
# own session, so it outlives a killed voicelab. It can also stop listening and still not exit, so
# looking at ports is not enough. The command line guards against a reused pid.
declare -A CHILDREN=()

command_of() { tr '\0' ' ' 2> /dev/null < "/proc/$1/cmdline"; }

note_children() {
  local parent pid
  for parent in "$SERVER_PID" $(pgrep -P "$SERVER_PID" 2> /dev/null); do
    for pid in $(pgrep -P "$parent" 2> /dev/null); do
      [ -n "${CHILDREN[$pid]:-}" ] || CHILDREN[$pid]="$(command_of "$pid")"
    done
  done
}

still_ours() { [ "$(command_of "$1")" = "${CHILDREN[$1]}" ]; }

signal_child() { # signal pid. Servers lead their own process group, and eval workers don't.
  kill "-$1" -- "-$2" 2> /dev/null || kill "-$1" "$2" 2> /dev/null
}

# Normally voicelab has stopped all of these by now.
stop_leftovers() {
  local pid _
  local -a left=()
  for pid in "${!CHILDREN[@]}"; do
    if still_ours "$pid"; then
      left+=("$pid")
      signal_child TERM "$pid"
    fi
  done
  [ "${#left[@]}" -gt 0 ] || return 0
  say "Stopping ${#left[@]} process(es) left behind."
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    for pid in "${left[@]}"; do
      still_ours "$pid" && break
      pid=""
    done
    [ -n "$pid" ] || return 0
    sleep 1
  done
  for pid in "${left[@]}"; do
    still_ours "$pid" && signal_child KILL "$pid"
  done
}

report_leftovers() {
  local port name owner pid clean=1
  while read -r port name; do
    [ -n "$port" ] || continue
    owner="$(port_owner "$port")"
    [ -z "$owner" ] || {
      say "Port $port ($name) is still held by $owner."
      clean=0
    }
  done <<< "$SERVICES"
  for pid in "${!CHILDREN[@]}"; do
    if still_ours "$pid"; then
      say "Process $pid is still running: ${CHILDREN[$pid]}"
      clean=0
    fi
  done
  [ -z "$SERVICES" ] || [ "$clean" = 0 ] || say "All model servers are stopped."
}

LAST_PROGRESS=""
PHASE=starting
STARTED=$SECONDS
while kill -0 "$SERVER_PID" 2> /dev/null; do
  if [ "$PHASE" != running ] || [ $((SECONDS % 5)) -eq 0 ]; then note_children; fi
  if [ "$STOP_ASKED" -gt 0 ]; then
    if [ "$SECONDS" -ge "$STOP_DEADLINE" ]; then
      say "Still running after $STOP_GRACE_S s, so forcing it."
      force_stop
    fi
  else
    case "$PHASE" in
      starting)
        if curl -fs -m 2 --noproxy '*' -o /dev/null "$BASE/api/health"; then
          print_urls
          PHASE=loading
          [ "$LAZY" = 0 ] || PHASE=running
          [ -n "$SERVICES" ] || PHASE=running
          STARTED=$SECONDS
        elif [ $((SECONDS - STARTED)) -ge "$HEALTH_LIMIT_S" ]; then
          say "No answer on $BASE after $HEALTH_LIMIT_S s. Still waiting. See ${LOG#"$ROOT"/}."
          STARTED=$SECONDS
        fi
        ;;
      loading)
        if report_models; then
          say "All models are ready."
          PHASE=running
        elif [ $((SECONDS - STARTED)) -ge "$MODEL_LIMIT_S" ]; then
          say "Models are still not ready after $MODEL_LIMIT_S s. The dashboard shows which role is stuck."
          PHASE=running
        fi
        ;;
    esac
  fi
  sleep 1 &
  wait $! 2> /dev/null
done

wait "$SERVER_PID" 2> /dev/null
SERVER_CODE=$?
trap - INT TERM HUP
stop_leftovers
report_leftovers

if [ "$STOP_ASKED" -eq 0 ]; then
  say "voicelab exited on its own with code $SERVER_CODE. The end of ${LOG#"$ROOT"/}:" >&2
  tail -n 20 "$LOG" >&2
  exit 1
fi
say "Stopped."
exit 0
