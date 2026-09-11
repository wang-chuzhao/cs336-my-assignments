import cs336_systems.benchmark
import torch

def timer(function,warmup=5,repeat=10):
    #warmup
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    times=[]
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)

    return np.median(times)