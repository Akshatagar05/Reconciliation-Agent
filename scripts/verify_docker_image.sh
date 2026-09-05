#!/usr/bin/env bash
# Stage 15 packaging check (ARCHITECTURE.md §14), updated by Stage 16 after
# the sentence-transformers cleanup.
#
# §14's "one command starts the full demo on a clean machine ... no
# cold-run download" originally had a named sub-requirement about the
# sentence-transformers embedding model being pre-baked. Stage 16 removed
# that dependency entirely (see README.md's Build Status and Dockerfile
# header comment): a full source search found no import of
# `sentence_transformers` anywhere in `src/` — Stage 9's fuzzy candidate
# retrieval hit its target (100% precision, 0% false-match rate) using
# `rapidfuzz` alone. With the dependency gone, there is nothing left to
# pre-bake for that item, and this script's checks were updated to match:
# it now proves the dependency really is gone from the built image (not
# just from requirements.txt) and that the datasets are still pre-baked,
# rather than checking for a model that's no longer supposed to be there.
#
#   1. Build the image.
#   2. Confirm `sentence_transformers` and `torch` are NOT importable
#      inside the built image — proving the cleanup actually took effect
#      at the image layer, not just in requirements.txt/pyproject.toml.
#   3. Confirm data/calibration and data/evaluation (plus their
#      ground_truth/ counterparts) are present, non-empty, and readable
#      in a throwaway container run with networking disabled
#      (`--network none`) — i.e. genuinely baked into the image, not
#      fetched or copied from the host at run time.
#
# Usage:  ./scripts/verify_docker_image.sh
# Requires: Docker (this script does not install it).
# Exit code 0 = every check passed. Any failure exits non-zero
# immediately (`set -euo pipefail`) with the failing command visible.

set -euo pipefail

IMAGE_TAG="recon-agent:verify"

echo "== [1/3] Building image (${IMAGE_TAG}) =="
docker build -t "${IMAGE_TAG}" .

echo
echo "== [2/3] Confirming sentence-transformers/torch are gone from the image =="
docker run --rm --network none "${IMAGE_TAG}" python -c "
import importlib.util, sys

removed = ['sentence_transformers', 'torch']
still_present = [name for name in removed if importlib.util.find_spec(name) is not None]
if still_present:
    print(f'FAIL: expected removed but still importable: {still_present}', file=sys.stderr)
    sys.exit(1)
print(f'OK: {removed} are not present in the image (Stage 16 cleanup confirmed)')
"

echo
echo "== [3/3] Confirming calibration/evaluation datasets are baked in =="
docker run --rm --network none "${IMAGE_TAG}" python -c "
import json, pathlib, sys

for path in [
    'data/calibration/records.json',
    'data/evaluation/records.json',
    'ground_truth/calibration/ground_truth.json',
    'ground_truth/evaluation/ground_truth.json',
]:
    p = pathlib.Path('/app') / path
    if not p.exists():
        print(f'FAIL: {path} missing from image', file=sys.stderr)
        sys.exit(1)
    data = json.loads(p.read_text())
    n = len(data) if isinstance(data, list) else len(data.get('match_groups', [])) + len(data.get('unresolved', [])) + len(data.get('duplicates', []))
    if n == 0:
        print(f'FAIL: {path} present but empty', file=sys.stderr)
        sys.exit(1)
    print(f'OK: {path} — {n} entries')
"

echo
echo "== All Docker packaging checks passed =="
