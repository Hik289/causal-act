#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_RUN_DIR}
echo "[$(date)] loading forum.tar"
docker load --input forum.tar
echo "LOAD_RC=$?"
: # Process and container cleanup is managed by the operator.
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
echo "RUN_RC=$?"
sleep 40
docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' | grep forum
curl -s -m 20 -o /dev/null -w 'SITE_HTTP=%{http_code}\n' http://127.0.0.1:9999/
echo DONE
