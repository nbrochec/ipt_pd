#!/bin/sh
# Headless test of ipt~ with libipt's dummy model (sr 24000, seglen 12000,
# 4 classes). Runs tests/ipt_test.pd in real time without an audio device for
# about 6 s, then checks what the [print] objects wrote.
#
#   tests/run_tests.sh [path/to/pd]
#
# Needs a build first (dist/ipt~). Real-time behaviour with a sound card is
# not covered here.

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
pd=${1:-$(ls -d /Applications/Pd-*.app 2>/dev/null | tail -n 1)/Contents/Resources/bin/pd}
log="$here/last_run.log"

if [ ! -x "$pd" ]; then
    echo "pd not found: $pd" >&2
    exit 2
fi
if [ ! -f "$root/dist/ipt~/ipt~.d_arm64" ]; then
    echo "build first: $root/dist/ipt~/ipt~.d_arm64 is missing" >&2
    exit 2
fi

# -noprefs: an ipt~ installed in the user's Pd paths must not shadow dist/
"$pd" -noprefs -nogui -noaudio -nomidi -stderr -path "$root/dist" "$here/ipt_test.pd" > "$log" 2>&1 &
pid=$!
# watchdog: kill pd after 40 s; exits by itself as soon as pd is gone
( i=0
  while kill -0 "$pid" 2>/dev/null; do
      i=$((i + 1))
      if [ "$i" -ge 400 ]; then kill -9 "$pid"; break; fi
      sleep 0.1
  done ) &
watchdog=$!
wait "$pid"
status=$?
wait "$watchdog"

failures=0
check() {  # check <description> <command...>
    desc=$1; shift
    if "$@"; then
        echo "PASS  $desc"
    else
        echo "FAIL  $desc"
        failures=$((failures + 1))
    fi
}
count() { grep -c "^$1" "$log"; }

check "pd exited normally (status $status)" test "$status" -eq 0
check "pd reached the end of the test" grep -q '^MARK-done: bang' "$log"
check "banner printed" grep -q 'ipt~ v.* for Pure Data' "$log"
check "model loaded (verbose instance)" grep -q 'ipt~: model loaded, 4 classes' "$log"

check "classnames: 4 names on the info outlet" \
    sh -c "grep '^A-info: classnames' '$log' | awk '{ exit !(NF == 6) }'"

check "A: distributions came out" test "$(count 'A-dist:')" -gt 10
check "A: every distribution has 4 values summing to 1" \
    sh -c "grep '^A-dist:' '$log' | awk '{ s = 0; for (i = 2; i <= NF; i++) s += \$i;
           if (NF != 5 || s < 0.99 || s > 1.01) bad = 1 } END { exit bad }'"
check "A: class index always in 0..3" \
    sh -c "grep '^A-index:' '$log' | awk '{ if (\$2 < 0 || \$2 > 3) bad = 1 } END { exit bad }'"
check "A: class name sent with each index" test "$(count 'A-name: symbol')" -eq "$(count 'A-index:')"
check "A: latency after each output" test "$(count 'A-info: latency')" -eq "$(count 'A-index:')"

check "B (confidence 1): outputs came out" test "$(count 'B-index:')" -gt 0
check "B (confidence 1): always -1" \
    sh -c "grep '^B-index:' '$log' | awk '{ if (\$2 != -1) bad = 1 } END { exit bad }'"
check "B (confidence 1): always no_confidence" \
    test "$(count 'B-name: symbol no_confidence')" -eq "$(count 'B-index:')"

check "C (silence, threshold -20): gated, no output" test "$(count 'C-index:')" -eq 0

d=$(count 'D-index:')
check "D (period 1000): 3 to 6 outputs in ~6 s (got $d)" sh -c "[ $d -ge 3 ] && [ $d -le 6 ]"

check "E (@enabled 0): nothing before 'enabled 1'" \
    sh -c "awk '/^MARK-enable/ { exit } /^E-index:/ { bad = 1 } END { exit bad }' '$log'"
check "E (@enabled 0): outputs after 'enabled 1'" \
    sh -c "sed -n '/^MARK-enable/,\$p' '$log' | grep -q '^E-index:'"

# mean absolute change between consecutive distributions of one instance
mean_step() {
    grep "^$1:" "$log" | awk '{ if (NR > 1) { for (i = 2; i <= NF; i++) { d = $i - p[i]; s += (d < 0 ? -d : d) } n++ }
                                for (i = 2; i <= NF; i++) p[i] = $i }
                              END { printf "%.6f", (n ? s / n : 0) }'
}
# total drift between the first and the last distribution
drift() {
    grep "^$1:" "$log" | awk 'NR == 1 { for (i = 2; i <= NF; i++) f[i] = $i } { for (i = 2; i <= NF; i++) l[i] = $i }
                              END { for (i in f) { d = l[i] - f[i]; s += (d < 0 ? -d : d) } printf "%.6f", s }'
}
a_step=$(mean_step A-dist); h_step=$(mean_step H-dist); h_drift=$(drift H-dist)
check "H (sensitivity 0): smoothed, step $h_step vs $a_step unsmoothed" \
    awk -v h="$h_step" -v a="$a_step" 'BEGIN { exit !(h > 0 && h < a / 5) }'
check "H (sensitivity 0): still moving over time (drift $h_drift)" \
    awk -v d="$h_drift" 'BEGIN { exit !(d > 0.001) }'

check "I: outputs before the runtime threshold change" \
    sh -c "grep -q '^I-index:' '$log'"
check "I (threshold 0 + window 50 at 2 s): gated after the change" \
    sh -c "sed -n '/^MARK-thr-check/,\$p' '$log' | grep -q '^I-index:' && exit 1 || exit 0"

# Apple GPU. Also makes "pd exited normally" cover quitting while an MPS
# worker is running (pd quit does not free the patches; see stop_all_workers).
check "J (MPS): outputs came out" test "$(count 'J-index:')" -gt 0

check "K ([ipt~ dummy]): found in the ipt~ folder, .ts added" \
    grep -q 'ipt~: model loaded, 4 classes (.*/dist/ipt~/dummy.ts)' "$log"
check "K ([ipt~ dummy]): outputs came out" test "$(count 'K-index:')" -gt 0

check "F: missing model refused at creation, with the folders searched" \
    grep -q 'model file not found: does_not_exist (looked in the patch folder .*/tests, in Pd.s search paths and in the ipt~ folder .*/dist/ipt~)' "$log"
check "G: deletion while loading and while running survived" grep -q '^MARK-clear2: bang' "$log"
check "no worker error reported" sh -c "! grep -q 'ipt~: model architecture\|ipt~: cannot initialise' '$log'"

echo
echo "log: $log"
if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
