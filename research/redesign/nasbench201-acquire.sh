#!/usr/bin/env bash
# Recreate the pinned NASBench201 source and data under one target directory.
set -euo pipefail

target="${1:?Pass a target directory with at least 4 GB free}"
mkdir -p "$target"
target="$(cd "$target" && pwd)"

python3 -m venv "$target/venv"
"$target/venv/bin/pip" install -q 'gdown==6.4.0'

if [[ ! -d "$target/evoxbench-source/.git" ]]; then
    git clone -q https://github.com/EMI-Group/evoxbench.git "$target/evoxbench-source"
fi
git -C "$target/evoxbench-source" checkout -q 2f1ae28720a09fdf3e487bcd9de91433512fc106
[[ "$(git -C "$target/evoxbench-source" rev-parse HEAD)" == \
    2f1ae28720a09fdf3e487bcd9de91433512fc106 ]]

if [[ ! -f "$target/database.zip" ]]; then
    "$target/venv/bin/gdown" 11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO \
        -O "$target/database.zip"
fi
if [[ ! -f "$target/data.zip" ]]; then
    "$target/venv/bin/gdown" 1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp \
        -O "$target/data.zip"
fi

(
    cd "$target"
    printf '%s\n' \
        'eb53451b37365517078bd1442ceba428e5649160394e6359e4c178eb07c8bf71  database.zip' \
        'e98557c120b8b4889443d9b9467c4b0fd80495797fbc533d1be2ba59f3264c40  data.zip' \
        | sha256sum -c -
    unzip -tqq database.zip
    unzip -tqq data.zip
    unzip -nq database.zip
    unzip -nq data.zip 'data20240229/data/nb201/*'
    printf '%s\n' \
        'd2dbd2eff43fdbc59dd3ec1873a92c98a1e1c88ecbeb4f0b62cff818e82921a5  database/db.sqlite3' \
        'd3307db64c76f8967ce3f8c9c1c5242860d326f7d30a6ae83ed87718308efe24  data20240229/data/nb201/nb201_pf.json' \
        'd671f16c25024c4bfb6780f58752c4728e8f1f036f23840a1eaecdae1c779ebc  data20240229/data/nb201/nb201_ps.json' \
        | sha256sum -c -
)

"$target/venv/bin/pip" install -q -e "$target/evoxbench-source"
printf 'Pinned EvoXBench data ready at %s\n' "$target"
