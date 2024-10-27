import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取CUDA设备数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # 遍历所有CUDA设备
    for i in range(num_gpus):
        # 获取并打印设备名称
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("CUDA is not available.")