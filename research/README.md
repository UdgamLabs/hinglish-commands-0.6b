# Reproduce the scientific experiment

These are unchanged scientific scripts, modules and selected prepared human datasets from the completed experiment. They are separate from the new developer wrapper in `src/`.

Read [Data and training](../docs/DATA_AND_TRAINING.md) and [Benchmarks](../docs/BENCHMARKS.md) before running anything. Training needs a compatible BF16 CUDA GPU and is not started by installation or tests.

From this directory, `scripts/restore_sources.py` checks the exact source manifest. Add `--download` to explicitly restore omitted public source files, including the unused synthetic source required by the historical full-manifest check. It does not download model weights. The original manifest remains unchanged. Prepared human data can be used without restoring the unused synthetic source.

`evidence/final-test/` contains the four saved prediction streams, complete score JSONs and paired comparisons. `evidence/file-identities.json` identifies their original byte content. These are the original results, not new SDK benchmarks. The original test freeze is retained in `evidence/test-freeze.json`; paths inside it refer to the original experiment layout. Reproduce into a new run directory, without overwriting stored evidence.

The full historical report included private operator paths and cloud-recovery details. Public docs preserve scientific methods, model/data identities, host-transition limitations, results and failure examples while excluding personal credentials and operational identifiers.
