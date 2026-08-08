import torch
import json

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

results = {}

# 1. Basic Indexing (ix-unit-examples-test)
t = torch.arange(120).float().reshape(2, 6, 10)
results["indexing"] = {
    "selection": tensor_to_list(t[1]),
    "slicing": tensor_to_list(t[:, 1:4, :]),
    "ellipsis": tensor_to_list(t[..., 0]),
    "open_step": tensor_to_list(t[:, :, ::2]),
    "negative": float(t[-1, -1, -1])
}

# 2. Advanced Indexing
data = torch.randn(3, 10)
indices = torch.tensor([0, 2, 0, 1])
results["advanced_indexing"] = tensor_to_list(data[indices, :])

# 3. Math Ops
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([4.0, 5.0, 6.0])
t3 = t1 + t2
t4 = t1 * 2.0
results["math_ops"] = {
    "add": tensor_to_list(t3),
    "mul": tensor_to_list(t4)
}

print(json.dumps(results))
