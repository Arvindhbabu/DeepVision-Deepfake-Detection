# DeepVision AI — Reproducibility Specification

## 1. Reproducibility Guarantee

DeepVision AI enforces scientific reproducibility across dataset splitting, model initialization, training execution, evaluation, and inference.

---

## 2. Key Reproducibility Mechanisms

1. **Seed Initialization (`set_seed`)**:
   - Python `random.seed(seed)`
   - NumPy `np.random.seed(seed)`
   - PyTorch `torch.manual_seed(seed)`
   - PyTorch CUDA `torch.cuda.manual_seed_all(seed)`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`

2. **Configuration-Driven Execution**:
   - All parameters (batch size, learning rate, sequence length, image size, FPS, optimizer, scheduler) are driven by YAML configurations in `configs/default.yaml`.
   - Active configuration YAML files are saved inside output run directories (`outputs/runs/run_*/config.yaml`).

3. **Environment Lock**:
   - Explicit dependency version locking in `requirements.txt` and `environment.yml`.
   - Verified automated CI test suite in `.github/workflows/ci.yml`.
