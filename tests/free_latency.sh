#!/bin/sh
# How long deleting an ipt~ blocks the Pd thread while its model is still
# loading, on CPU and on MPS, in a patch that is already running the same
# model. Needs a real model: dummy.ts loads in about 1 ms, too fast to show
# anything.
#
# Pd is quit 10 ms after the deletion, while the cleanup thread is still
# waiting for the load, which checks that exiting waits for it. The running
# instance makes torch create its lazy statics before the deletion, as in a
# patch that has been playing for a while; quitting during the very first
# load of a model kind is a known limit (see stop_all_workers in
# src/ipt_tilde.cpp).
#
#   tests/free_latency.sh /path/to/model.ts [path/to/pd]
#
# The model path must not contain spaces. Passes if each deletion blocks Pd
# for less than 20 ms and Pd exits cleanly.

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
model=${1:?usage: tests/free_latency.sh /path/to/model.ts [path/to/pd]}
pd=${2:-$(ls -d /Applications/Pd-*.app 2>/dev/null | tail -n 1)/Contents/Resources/bin/pd}
tmp=$(mktemp -d)
failures=0

for device in CPU MPS; do
    cat > "$tmp/free_$device.pd" <<EOF
#N canvas 0 0 300 200 12;
#X obj 10 10 loadbang;
#X obj 10 40 delay 3000;
#X obj 10 70 t b b b;
#X obj 10 130 realtime;
#X msg 50 100 \; pd-sub obj 10 10 ipt~ $model $device \; pd-sub clear;
#N canvas 0 0 300 200 sub 0;
#X restore 200 10 pd sub;
#X obj 10 160 print FREE-ms;
#X obj 100 40 delay 3010;
#X msg 100 70 \; pd quit;
#X msg 120 130 \; pd dsp 1;
#X obj 200 100 noise~;
#X obj 200 130 ipt~ $model $device;
#X obj 200 160 print RUNNING;
#X connect 0 0 1 0;
#X connect 1 0 2 0;
#X connect 2 2 3 0;
#X connect 2 1 4 0;
#X connect 2 0 3 1;
#X connect 3 0 6 0;
#X connect 0 0 7 0;
#X connect 7 0 8 0;
#X connect 0 0 9 0;
#X connect 10 0 11 0;
#X connect 11 0 12 0;
EOF
    "$pd" -noprefs -nogui -noaudio -nomidi -stderr -path "$root/dist" "$tmp/free_$device.pd" \
        > "$tmp/$device.log" 2>&1
    status=$?
    ms=$(grep '^FREE-ms:' "$tmp/$device.log" | awk '{ print $2 }')
    running=$(grep -c '^RUNNING:' "$tmp/$device.log")
    if [ "$status" -eq 0 ] && [ "$running" -gt 0 ] && [ -n "$ms" ] \
       && awk -v t="$ms" 'BEGIN { exit !(t < 20) }'; then
        echo "PASS  $device: deletion blocked Pd for $ms ms, clean exit"
    else
        echo "FAIL  $device: blocked ${ms:-?} ms, pd exit status $status, $running outputs from the running instance"
        grep -i error "$tmp/$device.log"
        failures=$((failures + 1))
    fi
done

rm -rf "$tmp"
exit "$failures"
