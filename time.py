import numpy as np
import network
import torch
from torch.backends import cudnn
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)

repetitions = 20

dummy_input = torch.rand(1, 3, 640, 640).to(device)

print(device)
# my_model.load_state_dict(torch.load("model/our_deblur80.pth"))
# my_model.load_state_dict(torch.load("model/4KDehazing.pth"))


# 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
print('warm up ...\n')
with torch.no_grad():
    for _ in range(10):
        _ = my_model(dummy_input)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
torch.cuda.synchronize()


# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
timings = np.zeros((repetitions, 1))


print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = my_model(dummy_input)
        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))