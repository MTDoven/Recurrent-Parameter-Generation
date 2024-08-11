from pynvml import *
import torch
import time


nvmlInit()
memory_threshold = 75000  # MiB
control_cpu_index = [0, 1, 2, 3]
total_memory_byte = 81920 * 1024 * 1024


useless_tensor = {
    0: torch.zeros(size=(0,), dtype=torch.uint8).cuda(0),
    1: torch.zeros(size=(0,), dtype=torch.uint8).cuda(1),
    2: torch.zeros(size=(0,), dtype=torch.uint8).cuda(2),
    3: torch.zeros(size=(0,), dtype=torch.uint8).cuda(3),
}

try:
    memory_threshold_byte = memory_threshold * 1024 * 1024
    while True:
        for i in control_cpu_index:
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            free_memory = mem_info.free
            if free_memory > total_memory_byte - memory_threshold_byte:  # to allocate
                useless_tensor[i] = torch.zeros(
                    size=(free_memory - (total_memory_byte - memory_threshold_byte) + useless_tensor[i].numel(),),
                    dtype=torch.uint8,
                ).cuda(i)
            elif free_memory < total_memory_byte - memory_threshold_byte - 3 * 1024 * 1024:
                useless_tensor[i] = torch.zeros(
                    size=(0,),
                    dtype=torch.uint8,
                ).cuda(i)
                torch.cuda.empty_cache()

            print("\r", end="")
            for k, v in useless_tensor.items():
                print("cuda"+str(k)+":", v.numel() / (1024 * 1024), "MiB  |  ", end="")
            time.sleep(0.1)

except NVMLError as error:
    print(f"NVIDIA Management Library error: {error}")

finally:
    nvmlShutdown()