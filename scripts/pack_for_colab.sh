#!/bin/sh
# Pack code + token binaries + eval sets for Colab. Upload the result to Drive as MyDrive/trlm/trlm_colab.zip
set -e
cd "$(dirname "$0")/.."
rm -f trlm_colab.zip
zip -r trlm_colab.zip src configs data/tokens/train.bin data/eval data/tokenizer.json data/prep_meta.json \
    -x '*/__pycache__/*'
ls -la trlm_colab.zip
