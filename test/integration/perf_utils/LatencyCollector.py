import time
import math
import os
import json

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

def benchmark(n_runs, test_name, model, model_inputs, metric_path="."):
    # model inputs can be tuple or dictionary
    if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, dict):
        model_inputs = (model_inputs,)

    def run_model():
        if isinstance(model_inputs, dict):
            return model(**model_inputs)
        else : #tuple
            return model(*model_inputs)

    warmup_run = run_model()

    latency_collector = LatencyCollector()
    # can't use register_forward_pre_hook or register_forward_hook because StableDiffusionPipeline is not a torch.nn.Module
    
    for _ in range(n_runs):
        latency_collector.pre_hook()
        res = run_model()
        latency_collector.hook()
    
    p0_latency_ms = latency_collector.percentile(0) * 1000
    p50_latency_ms = latency_collector.percentile(50) * 1000
    p90_latency_ms = latency_collector.percentile(90) * 1000
    p95_latency_ms = latency_collector.percentile(95) * 1000
    p99_latency_ms = latency_collector.percentile(99) * 1000
    p100_latency_ms = latency_collector.percentile(100) * 1000

    report_dict = dict()
    report_dict["Latency.P0"] = f'{p0_latency_ms:.1f}'
    report_dict["Latency.P50"]=f'{p50_latency_ms:.1f}'
    report_dict["Latency.P90"]=f'{p90_latency_ms:.1f}'
    report_dict["Latency.P95"]=f'{p95_latency_ms:.1f}'
    report_dict["Latency.P99"]=f'{p99_latency_ms:.1f}'
    report_dict["Latency.P100"]=f'{p100_latency_ms:.1f}'

    report = f'RESULT FOR {test_name}:'
    for key, value in report_dict.items():
        report += f' {key}={value}'
    print(report)

    report = {"test_name": test_name, "report": report_dict}
    
    with open(metric_path, 'r+') as f:
        cur = json.load(f)
        cur.append(report)
    with open(metric_path, 'w+') as f:
        json.dump(cur, f)
    return report_dict
