# 🚫 PROTOCOL: ANTI-CONTEXT-ASSASSINS

## Objective
Prevent AI models (Gemini/Antigravity) from attempting to read heavy binary or massive text files that cause "Cortex Errors" or context pollution.

## Mandatory Exclusions (Never Read/Import)
The following files are **OFF-LIMITS** for AI agents:

### 1. AI Models & Weights (.pth, .pt, .joblib, .h5, .onnx, .pkl)
- These are numerical matrices, not human-readable code.
- Examples: `models/feath_hybrid_v1.pth`, `models/gbm_XAUUSD_v1.joblib`.

### 2. Databases & Storage (.db, .sqlite3, .csv, .jsonl)
- Large datasets or binary databases.
- Folders: `chroma_storage/`, `data/`.

### 3. Massive Noise Generators
- `package-lock.json`: Contains tens of thousands of lines of dependency metadata.
- `logs/*.log`: System logs that grow too large for context windows.

## Enforcement Rule for Antigravity
> [!IMPORTANT]
> **Antigravity Rule**: If you detect any file matching the above patterns, DO NOT use `view_file` or `view_file_outline` on it. Skip it during any codebase audit.

## Verification
The `.gitignore` has been updated to reflect these rules and the files have been physically removed from the current environment to ensure a clean "Code-Only" workspace.
