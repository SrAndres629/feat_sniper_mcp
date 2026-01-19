# Skill: VERIFICATOR SENTINEL (Integrity & Debugging)

## 🛡️ Description
Expert protocol for continuous validation, debugging, and integrity checking of a high-frequency trading system. Its mission is to ensure that no refactoring or architectural change collapses the internal dependencies or the compilation logic of the "Immortal Core".

## 🛠️ Verification Workflow

### 1. Pre-Flight Integrity Check
- **Context Audit**: Before modifying, run `python audit_lines.py` (or check manually) and `python -m py_compile [file]` to ensure the starting state is valid.
- **Auto-Refactor Trigger**: If the file exceeds **300 lines**, the Verificator MUST immediately invoke [ATOMIC FISSION](./refactor_fission.md) to modularize the file before any other operation.
- **Dependency Map**: Identify who imports the target file using `rg "from [module] import"`.

### 2. Autonomous Test Generation
For every refactored package, a `tests/verify_[package_name].py` MUST be created.
The test must cover:
- **Import Integrity**: `from [package] import [main_objects]`
- **Instantiation**: Verify class constructors with mock data.
- **Relative Path Check**: Ensure submodules can find each other inside the package.
- **Data Flow**: Verify that the output (Dictionary or Tensor) matches the expected schema for the Neural Network.

### 3. Real-Time Debugging Protocol
If a smoke test fails:
- **Traceback Analysis**: Read the LAST 20 lines of the error.
- **Isolation**: Test submodules individually (e.g., `python -f [submodule].py`) to find the exact point of failure.
- **Circular Dep Check**: Look for recursive imports introduced during fission.

### 4. Compilation & Build Validation
- **Global Check**: Periodically run `python nexus_daemon.py --check-only` (if implemented) or attempt to initialize the `NexusEngine` in a dry-run.
- **Type Checking**: Use `mypy` or similar if the user requests strict typing verification.

## 🎯 Success Condition
Zero (0) ImportErrors and 100% logic pass in the corresponding smoke test. No commit is allowed without a "Green Light" from the Verificator Sentinel.
