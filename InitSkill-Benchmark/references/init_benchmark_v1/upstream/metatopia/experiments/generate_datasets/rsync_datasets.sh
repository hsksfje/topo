#!/bin/bash

DATA_DIR="$(cd "$(dirname "$0")/../../data/to" && pwd)"
REMOTE="igor.kuszczak@thor.supsi.ch:scratch/metatopia/data/to/"

echo "Syncing test_cr_ood..."
rsync -avP "$DATA_DIR/test_cr_ood/" "$REMOTE/test_cr_ood/"
rsync -avP "$DATA_DIR/test_cr_ood.yaml" "$REMOTE/test_cr_ood.yaml"

echo "Syncing test_ar..."
rsync -avP "$DATA_DIR/test_ar/" "$REMOTE/test_ar/"
rsync -avP "$DATA_DIR/test_ar.yaml" "$REMOTE/test_ar.yaml"

echo "Done."
