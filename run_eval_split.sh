#!/bin/bash
cd ~/verantyx_v6
> arc_cross_engine_v23.log
for offset in $(seq 0 100 900); do
    limit=$((offset + 100))
    python3 -u -m arc.eval_cross_engine --split training --offset $offset --limit 100 >> arc_cross_engine_v23.log 2>&1
    echo "=== Batch $offset done ===" >> arc_cross_engine_v23.log
done
