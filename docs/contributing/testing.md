# Testing

RL-Kernel uses focused tests for dispatch behavior and operator accuracy.

## gtest (operator candidate vs gold)

Primary entry for single-operator forward/backward checks against a PyTorch gold path:

```bash
python scripts/check_operator.py --op logp --candidate pytorch --device cpu --dtype fp32 \
  --batch 1 --seq 2 --vocab 17
```

Full usage (register `OP_SPECS`, build inputs, CLI flags, and the WS1 four-judgment
tolerance contract after #267):

- **[gtest usage guide](gtest-usage.md)** (operator CLI + `OP_SPECS` + contract; English)

## Dispatch Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
```

## Operator Accuracy

```bash
python tests/test_op_accuracy.py
```

Contract schema / resolver and WS1 C1–C8 CPU gates:

```bash
python -m pytest tests/test_tolerance_contract.py tests/test_op_checks.py \
  tests/test_ws1_workload.py tests/test_forward_invariance.py \
  tests/test_gradient_invariance.py tests/test_elementwise_inventory.py \
  tests/test_four_judgment_matrix.py tests/test_operator_inputs.py -q
```

`max_abs_dlogp`, `approx_kl0`, and `clipfrac0` are the sole chain-level logprob
aggregates; gradient pass/fail uses independent `gradient_*` verdicts.

CUDA BF16 and Triton-on-CUDA BF16 gtest + C8 `--execute` run in
`.github/workflows/ws1-gtest-gpu.yml` (RunPod). Local equivalent:

```bash
bash ci/run_ws1_gtest.sh
```

The C8 JSON is written outside the repo (`${TMPDIR:-/tmp}/ws1-c8-ci.json` unless
`WS1_C8_JSON` is set) so the recorded git provenance is not dirtied by the
artifact itself. The GitHub workflow uploads that file as a CI artifact.

## Documentation Build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict -f mkdocs.yaml
```

Run the documentation build whenever adding a new operator page or changing navigation.
