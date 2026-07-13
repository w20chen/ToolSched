$ErrorActionPreference = "Stop"
$datasets = "C:\Users\29068\Desktop\agent_datasets"
python -m toolsched.cli inspect --datasets $datasets
python -m toolsched.cli build --datasets $datasets --out artifacts\samples.small.jsonl --limit-attempts 50
python -m toolsched.cli evaluate --samples artifacts\samples.small.jsonl --out artifacts\metrics.small.json
python -m toolsched.cli calibrate --samples artifacts\samples.small.jsonl --out artifacts\calibration.small.json
