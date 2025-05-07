#!/bin/bash
# wait-for-elastic.sh

set -e

host="$1"
shift
cmd="$@"

echo "Waiting for Elasticsearch at $host..."
until curl -s -f "$host/_cluster/health" > /dev/null; do
  >&2 echo "Elasticsearch is unavailable - sleeping 5s"
  sleep 5
done

>&2 echo "Elasticsearch is up - executing command"
exec $cmd