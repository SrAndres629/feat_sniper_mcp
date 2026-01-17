# 🗡️ Skill: Neural Surgeon

## 🎯 Objective
Surgical modification of the `HybridProbabilistic` model and synaptic tensor flows.

## 📜 Neural Safety Standards
1. **Dimension Strictness**: Any change to `forward()` MUST documented the expected input/output tensor shapes. (e.g., `(Batch, 1, 50, 50) -> (Batch, 32)`).
2. **Deterministic Veto**: Never bypass the `force_dropout` logic during inference. Production mode requires strictly $N=30$ MC iterations.
3. **Loss Integrity**: Retraining MUST use `ConvergentSingularityLoss` to ensure physics-aware gradients.

## 🛠️ Modus Operandi
- Verify `input_size` against `settings.NEURAL_INPUT_DIM` before saving.
- Check that Attention heads are correctly normalized with `Softmax(dim=1)`.
- Ensure Latent Fusion (`context + z_t + z_s`) does not introduce data-loss through incorrect concatenation.

## ✅ Success Criteria
- Synaptic test passes with zero `ShapeMismatch` errors.
- Epistemic uncertainty ($\sigma$) correctly correlates with market turbulence.
