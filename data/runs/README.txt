REAL BERNESE OUTPUT DATA
========================

This directory holds the actual Bernese session outputs used for the experiment.
Populate it with src/extract.py:

    python src/extract.py --prc path/to/RNX2SNX.PRC --crd path/to/RNX2SNX.CRD \
           --out data/runs/F0 --run-id F0

Then fill in data/runs/F0/truth.json with the correct action and your reasoning.

No directory should appear here until real Bernese runs are complete.
