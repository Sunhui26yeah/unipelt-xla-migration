"""Run run_glue.py exactly as `python run_glue.py ...` would (its __main__ block, including choose_gpu), but print
every Trainer.log() dict as one JSON line ("LOGJSON {...}"). On XLA it adds the CompileTime count and the aten::
CPU-fallback counters to each line, and writes torch_xla's metrics report to <output_dir>/xla_metrics_report.txt
at exit. Measurement-only harness code: it changes no computation.
Usage: python log_wrapper.py <run_glue.py args...>   (run from the code directory)
"""
import json
import os
import runpy
import sys
import time

sys.path.insert(0, os.getcwd())
import transformers.trainer as hf_trainer  # noqa: E402

_orig_log = hf_trainer.Trainer.log
_instances = []


def _xla_stats():
    if "torch_xla" not in sys.modules:
        return {}
    import torch_xla.debug.metrics as met
    d = met.metric_data("CompileTime")
    out = {"xla_compiles": d[0] if d else 0}
    out["aten_fallbacks"] = {c: met.counter_value(c) for c in met.counter_names() if c.startswith("aten::")}
    return out


def _log(self, logs):
    if self not in _instances:
        _instances.append(self)
    rec = {"global_step": self.state.global_step, "epoch": self.state.epoch, "time": time.time()}
    for k, v in logs.items():
        try:
            rec[k] = float(v) if not isinstance(v, (list, dict, str, bool)) else v
        except Exception:
            rec[k] = str(v)
    rec.update(_xla_stats())
    print("LOGJSON " + json.dumps(rec), flush=True)
    return _orig_log(self, logs)


hf_trainer.Trainer.log = _log

if __name__ == "__main__":
    t_start = time.time()
    sys.argv = ["run_glue.py"] + sys.argv[1:]
    try:
        runpy.run_path("run_glue.py", run_name="__main__")
    finally:
        for tr in _instances:
            try:
                with open(os.path.join(tr.args.output_dir, "log_history.json"), "w") as f:
                    json.dump(tr.state.log_history, f, default=str)
                if "torch_xla" in sys.modules:
                    import torch_xla.debug.metrics as met
                    with open(os.path.join(tr.args.output_dir, "xla_metrics_report.txt"), "w") as f:
                        f.write(met.metrics_report())
            except Exception as e:
                print("LOGJSON_DUMP_FAILED", e)
        print(f"WRAPPER_TOTAL_SECONDS {time.time() - t_start:.2f}", flush=True)
