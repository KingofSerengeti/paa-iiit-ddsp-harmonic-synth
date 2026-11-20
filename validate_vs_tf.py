"""
Validate ddsp_port.py against the REAL TensorFlow DDSP core.py (CPU only).
Three steps, because TensorFlow and PyTorch should live in different virtualenvs:

  1) torch env :  python validate_vs_tf.py gen                 -> writes val_inputs.npz and val_port.npz
  2) tf env    :  python validate_vs_tf.py tf /path/to/ddsp/ddsp/core.py   -> writes val_tf.npz
  3) torch env :  python validate_vs_tf.py compare

tf env setup:  python3 -m venv ~/tf-env && source ~/tf-env/bin/activate && pip install tensorflow-cpu gin-config scipy numpy
(core.py is loaded directly from its file, so the rest of the ddsp package and its heavy dependencies are not needed.)
"""
import sys, importlib.util
import numpy as np

B, K, T, SR, HOP = 2, 100, 16000, 16000, 64

def gen():
    import torch
    from ddsp_port import make_controls, harmonic_synthesis
    f0, amp, dist = make_controls(B, K, T // HOP, SR, device="cpu")
    np.savez("val_inputs.npz", f0=f0.numpy(), amp=amp.numpy(), dist=dist.numpy())
    out = {}
    for ang in (False, True):
        o32 = harmonic_synthesis(f0, amp, dist, T, SR, "window", ang).numpy()
        o64 = harmonic_synthesis(f0.double(), amp.double(), dist.double(), T, SR, "window", False).numpy()
        out[f"port32_ang{int(ang)}"] = o32
        out["port64"] = o64
    np.savez("val_port.npz", **out); print("wrote val_inputs.npz, val_port.npz")

def tf_run(core_path):
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    spec = importlib.util.spec_from_file_location("ddsp_core", core_path)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    d = np.load("val_inputs.npz"); out = {}
    for ang in (False, True):
        a = core.harmonic_synthesis(frequencies=d["f0"], amplitudes=d["amp"], harmonic_distribution=d["dist"],
                                    n_samples=T, sample_rate=SR, amp_resample_method="window",
                                    use_angular_cumsum=ang)
        out[f"tf_ang{int(ang)}"] = np.asarray(a)
    np.savez("val_tf.npz", **out); print("wrote val_tf.npz")

def compare():
    p, t = np.load("val_port.npz"), np.load("val_tf.npz")
    ref = p["port64"].astype(np.float64)
    def stats(a, b): return f"max_abs={np.abs(a-b).max():.3e}  rel_L2={np.linalg.norm(a-b)/np.linalg.norm(b):.3e}"
    for ang in (0, 1):
        tf_o = t[f"tf_ang{ang}"].astype(np.float64); po = p[f"port32_ang{ang}"].astype(np.float64)
        print(f"angular_cumsum={bool(ang)}")
        print("  TF fp32   vs fp64 truth :", stats(tf_o, ref))
        print("  port fp32 vs fp64 truth :", stats(po, ref))
        print("  port fp32 vs TF fp32    :", stats(po, tf_o))
    print("\nGood sign: port-vs-TF error is about the same size as each one's error vs the fp64 truth (pure rounding noise).")
    print("A much larger port-vs-TF error means a real implementation difference to fix.")

if __name__ == "__main__":
    {"gen": gen, "tf": lambda: tf_run(sys.argv[2]), "compare": compare}[sys.argv[1]]()
