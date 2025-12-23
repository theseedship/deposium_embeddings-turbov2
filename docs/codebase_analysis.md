# Codebase Analysis and Quick Wins

## Scope
This review focuses on surfacing low-effort cleanups, especially unused code paths or files that can be removed or archived safely.

## Quick Wins

1. **Remove unused local benchmarking script**
   - `benchmark_embeddings.py` is not imported or referenced anywhere in the repository. A ripgrep scan for the filename and primary function names returns no matches, indicating it is effectively dead code. Removing it trims maintenance surface and avoids confusion about supported benchmarking paths.【F:benchmark_embeddings.py†L1-L44】【20029b†L1-L2】

2. **Prune orphaned student model implementation**
   - `src/models/student_model.py` defines the `LEAFStudent` architecture and helper creation utilities but is not referenced outside the file. No other module imports this class, so it ships unused training/inference logic plus a large set of heavyweight transformer dependencies for a feature that is not wired into the service. Archiving or deleting the module would reduce cognitive load and potential dependency drift.【F:src/models/student_model.py†L1-L98】【8f4392†L1-L4】

3. **Drop unused model inspection helper**
   - `inspect_model.py` is a standalone diagnostic script that never gets imported. It pulls transformer configs at runtime and adds no runtime value to the service. Removing or moving it to a separate ops repository would reduce image size and avoid accidental execution in production builds.【F:inspect_model.py†L1-L44】【a931c9†L1-L1】

4. **Validate necessity of OpenBench integration**
   - OpenBench endpoints are exposed (`/api/benchmarks*`), but the only usage of `OpenBenchRunner` is inside these handlers. There is no internal caller, and the feature requires the optional `openbench` dependency. If benchmarking is not part of the deployed surface, disabling the endpoints and dropping the dependency would slim deployments and avoid runtime errors when `openbench` is unavailable. Conversely, if kept, adding automated coverage is advisable because the code path currently has no referenced consumers.【F:src/main.py†L520-L607】【F:src/benchmarks/openbench_runner.py†L16-L89】【667c36†L1-L11】

5. **Confirm necessity of Gemma and Qwen3 embedding configs**
   - `model_manager.py` registers several embedding models (`gemma-768d`, `qwen3-embed`, `bge-m3-onnx`) alongside the primary `m2v-bge-m3-1024d`. There are no dedicated endpoints exercising the non-default models beyond the generic `/api/embed` handler, and only `m2v-bge-m3-1024d` is highlighted as primary. If these legacy/experimental configs are unused in production, pruning them will simplify model lifecycle management and reduce VRAM bookkeeping complexity.【F:src/model_manager.py†L54-L122】【F:src/main.py†L58-L115】

## Notes on Verification
- Unused-code findings were double-checked with repository-wide searches (no matches beyond the defining files). Wherever removal is recommended, run `rg <symbol>` after deletion to ensure no hidden references remain and adjust Dockerfiles or documentation if the files are pruned.
- Before removing optional dependencies like `openbench`, confirm whether downstream automation relies on them; if not, they can be moved to a dev-only extra.
