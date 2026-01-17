# Skill: ATOMIC FISSION (Modular Refactoring)

## Description
Protocol for breaking down monolithic "God Objects" (Python files > 300 lines) into granular, high-cohesion submodules without breaking existing system dependencies.

## Algorithm of Action

### 1. Dependency Mapping
- Scan the project for files that import the target module.
- Identify public APIs (classes, functions, constants) that must remain accessible.

### 2. Package Transformation
- Create a directory with the same name as the original `target_file.py` (minus the `.py`).
- Create an `__init__.py` file in the new directory.

### 3. Segregation Logic
- Divide the original content into logical thematic files (e.g., `logic.py`, `models.py`, `utils.py`).
- Use the following naming convention for `structure_engine`:
    - `fractals.py`: Pivot and fractal detection math.
    - `zones.py`: Supply/Demand zone logic.
    - `transitions.py`: CHOCH, BOS, and structural shifts.

### 4. Interface Exposure (Backward Compatibility)
- Import the main classes/functions into `__init__.py`.
- Ensure that `from target_file import TargetClass` still works by using `from .submodule import TargetClass` inside `__init__.py`.

### 5. Hot-Wiring & Verification
- Update internal imports within the submodules (use relative imports like `from .fractals import ...`).
- Run a syntax check to ensure no circular dependencies.
- Delete the original monolithic file once verified.

## Goal
Achieve < 300 lines per file for maximum AI cognitive clarity and zero-entropy maintenance.
