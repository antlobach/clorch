import subprocess
import json
import torch
import numpy as np
import sys


def run_clorch(clojure_code):
    """
    Runs Clojure code using the Clojure CLI and returns the result (assumed to be JSON).
    """
    full_code = f"""
    (require '[clorch.torch :as torch])
    (require '[clorch.nn :as nn])
    (require '[clorch.nn.functional :as F])
    (require '[clorch.optim :as optim])
    (require '[clorch.autograd :as autograd])
    (require '[clorch.data :as data])
    (require '[clorch.comparison-utils :as utils])
    
    (torch/with-torch
      (torch/manual-seed 42)
      (let [result (do {clojure_code})]
        (utils/export-result result)))
    """
    process = subprocess.Popen(
        ["clojure", "-M", "-e", full_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Clojure error:\n{stderr}")
        return None

    # Extract the last line which should be the JSON
    lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from last line: {last_line}")
        print(f"Full stdout: {stdout}")
        return None


def compare_tensors(clj_t, py_t, name):
    # clj_t is a dict with shape, dtype, data
    # py_t is a torch.Tensor
    py_data = py_t.detach().cpu().numpy().flatten().tolist()
    py_shape = list(py_t.shape)

    # Map PyTorch dtypes to string descriptions seen in clorch
    # clorch uses (.toString stype) which returns "Float", "Long", etc.
    dtype_map = {
        torch.float32: "Float",
        torch.float64: "Double",
        torch.int32: "Int",
        torch.int64: "Long",
        torch.uint8: "Byte",
        torch.int8: "Char",
        torch.bool: "Bool",
    }

    if clj_t["shape"] != py_shape:
        print(
            f"FAILED {name}: Shape mismatch! Clorch: {clj_t['shape']}, PyTorch: {py_shape}"
        )
        return False

    # Check if clj_t['dtype'] starts with the expected type name (some presets add extra info)
    expected_dtype = dtype_map.get(py_t.dtype, "unknown")
    if not clj_t["dtype"].startswith(expected_dtype):
        print(
            f"FAILED {name}: Dtype mismatch! Clorch: {clj_t['dtype']}, PyTorch: {expected_dtype}"
        )
        return False

    # Convert data back to float, handling Inf/NaN strings
    def clean_val(x):
        if x == "Infinity":
            return float("inf")
        if x == "-Infinity":
            return float("-inf")
        if x == "NaN":
            return float("nan")
        return x

    clj_data = [clean_val(x) for x in clj_t["data"]]

    # Numerical comparison
    if not np.allclose(clj_data, py_data, atol=1e-5, equal_nan=True):
        print(f"FAILED {name}: Numerical mismatch!")
        print(f"  Clorch shape: {clj_t['shape']}, len(data): {len(clj_data)}")
        print(f"  PyTorch shape: {py_shape}, len(data): {len(py_data)}")
        # Print first 10
        print(f"  Clorch first 10: {clj_data[:10]}")
        print(f"  PyTorch first 10: {py_data[:10]}")
        return False

    print(f"PASSED {name}")
    return True


def run_tests():
    print("Starting Comparison Tests...")
    torch.manual_seed(42)

    # Pre-generate weights for parity tests
    torch.save(torch.randn(10, 3), "emb_weights.pt")

    all_passed = True

    def check_res(res, name):
        if res is None:
            print(f"FAILED {name}: Clojure returned None (likely an error)")
            return False
        return True

    # Test 1: Basic Creation & Arithmetic
    print("\n--- Basic Ops ---")
    clj_res = run_clorch("""
        (let [t1 (torch/tensor [1.0 2.0 3.0])
              t2 (torch/tensor [4.0 5.0 6.0])]
          {:add (torch/add t1 t2)
           :sub (torch/sub t1 t2)
           :mul (torch/mul t1 t2)
           :div (torch/div t1 t2)})
    """)

    if check_res(clj_res, "Basic Ops"):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([4.0, 5.0, 6.0])
        all_passed &= compare_tensors(clj_res["add"], t1 + t2, "add")
        all_passed &= compare_tensors(clj_res["sub"], t1 - t2, "sub")
        all_passed &= compare_tensors(clj_res["mul"], t1 * t2, "mul")
        all_passed &= compare_tensors(clj_res["div"], t1 / t2, "div")
    else:
        all_passed = False

    # Test 2: Factory Methods
    print("\n--- Factory Methods ---")
    clj_res = run_clorch("""
        {:ones (torch/ones [2 2])
         :zeros (torch/zeros [3 1])
         :eye (torch/eye 3)
         :randint (torch/rand-int 0 10 [2 2])}
    """)
    if check_res(clj_res, "Factory Methods"):
        all_passed &= compare_tensors(clj_res["ones"], torch.ones(2, 2), "ones")
        all_passed &= compare_tensors(clj_res["zeros"], torch.zeros(3, 1), "zeros")
        all_passed &= compare_tensors(clj_res["eye"], torch.eye(3), "eye")
        if clj_res["randint"]["shape"] == [2, 2] and all(
            0 <= x < 10 for x in clj_res["randint"]["data"]
        ):
            print("PASSED rand-int shape and range")
        else:
            print(f"FAILED rand-int: {clj_res['randint']}")
            all_passed = False
    else:
        all_passed = False

    # Test 3: Math Reductions & Matrix Ops
    print("\n--- Math & Matrix Ops ---")
    clj_res = run_clorch("""
        (let [t (torch/tensor [[1.0 2.0] [3.0 4.0]])]
          {:sum (torch/sum t)
           :mean (torch/mean t)
           :max (torch/max t)
           :min (torch/min t)
           :matmul (torch/matmul t t)
           :transpose (torch/transpose t 0 1)})
    """)
    if check_res(clj_res, "Math Ops"):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        all_passed &= compare_tensors(clj_res["sum"], torch.sum(t), "sum")
        all_passed &= compare_tensors(clj_res["mean"], torch.mean(t), "mean")
        all_passed &= compare_tensors(clj_res["max"], torch.max(t), "max")
        all_passed &= compare_tensors(clj_res["min"], torch.min(t), "min")
        all_passed &= compare_tensors(clj_res["matmul"], torch.matmul(t, t), "matmul")
        all_passed &= compare_tensors(
            clj_res["transpose"], torch.transpose(t, 0, 1), "transpose"
        )
    else:
        all_passed = False

    # Test 6: Optimizers (SGD & Adam)
    print("\n--- Optimizers (SGD & Adam) ---")
    clj_res = run_clorch("""
        (let [w1 (torch/tensor [10.0] {:requires-grad true})
              opt1 (optim/sgd [w1] {:lr 0.1 :momentum 0.9})
              w2 (torch/tensor [10.0] {:requires-grad true})
              opt2 (optim/adam [w2] {:lr 0.1 :weight-decay 0.01 :amsgrad true})]
          (dotimes [_ 5]
            (optim/zero-grad opt1)
            (autograd/backward (torch/mul w1 w1))
            (optim/step opt1)
            (optim/zero-grad opt2)
            (autograd/backward (torch/mul w2 w2))
            (optim/step opt2))
          {:sgd w1 :adam w2})
    """)
    if check_res(clj_res, "Optimizers"):
        w1 = torch.tensor([10.0], requires_grad=True)
        opt1 = torch.optim.SGD([w1], lr=0.1, momentum=0.9)
        w2 = torch.tensor([10.0], requires_grad=True)
        opt2 = torch.optim.Adam([w2], lr=0.1, weight_decay=0.01, amsgrad=True)
        for _ in range(5):
            opt1.zero_grad()
            (w1 * w1).backward()
            opt1.step()
            opt2.zero_grad()
            (w2 * w2).backward()
            opt2.step()
        all_passed &= compare_tensors(clj_res["sgd"], w1, "sgd-momentum-update")
        all_passed &= compare_tensors(clj_res["adam"], w2, "adam-custom-update")
    else:
        all_passed = False

    # Test 7: NN Layers & Loss
    print("\n--- NN Layers & Loss ---")
    clj_res = run_clorch("""
        (let [pred (torch/tensor [[0.1 0.2 0.7]])
              target (torch/tensor [2] {:dtype :int64})
              mse-pred (torch/tensor [0.5 0.5])
              mse-target (torch/tensor [1.0 0.0])]
          {:mse (F/mse-loss mse-pred mse-target)
           :ce (F/cross-entropy pred target)
           :relu (F/relu (torch/tensor [-1.0 0.0 1.0]))
           :sigmoid (F/sigmoid (torch/tensor [0.0]))
           :flatten (nn/forward (nn/flatten) (torch/ones [2 3 4]))
           :conv (nn/forward (nn/conv2d 1 1 3) (torch/ones [1 1 5 5]))
           :pool (nn/forward (nn/max-pool2d 2) (torch/ones [1 1 8 8]))})
    """)
    if check_res(clj_res, "NN Layers"):
        all_passed &= compare_tensors(
            clj_res["mse"],
            torch.nn.functional.mse_loss(
                torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0])
            ),
            "mse-loss",
        )
        all_passed &= compare_tensors(
            clj_res["ce"],
            torch.nn.functional.cross_entropy(
                torch.tensor([[0.1, 0.2, 0.7]]), torch.tensor([2])
            ),
            "cross-entropy",
        )
        all_passed &= compare_tensors(
            clj_res["relu"],
            torch.nn.functional.relu(torch.tensor([-1.0, 0.0, 1.0])),
            "relu",
        )
        all_passed &= compare_tensors(
            clj_res["sigmoid"], torch.sigmoid(torch.tensor([0.0])), "sigmoid"
        )
        all_passed &= compare_tensors(
            clj_res["flatten"], torch.flatten(torch.ones(2, 3, 4), 1), "flatten"
        )
        if clj_res["conv"]["shape"] == [1, 1, 3, 3]:
            print("PASSED conv2d shape")
        else:
            print(f"FAILED conv2d shape: {clj_res['conv']['shape']}")
            all_passed = False
        if clj_res["pool"]["shape"] == [1, 1, 4, 4]:
            print("PASSED max-pool2d shape")
        else:
            print(f"FAILED max-pool2d shape: {clj_res['pool']['shape']}")
            all_passed = False
    else:
        all_passed = False

    # Test 8: Dataloader
    print("\n--- Dataloader ---")
    clj_res = run_clorch("""
        (let [x (torch/randn [10 5])
              y (torch/randn [10 1])
              ds (data/tensor-dataset x y)
              dl (data/dataloader ds {:batch-size 4 :shuffle? false})]
          (vec (map (fn [b] {:data (:data b) :target (:target b)}) dl)))
    """)
    if check_res(clj_res, "Dataloader"):
        if len(clj_res) == 3:
            print(f"PASSED dataloader batch count: {len(clj_res)}")
            if clj_res[0]["data"]["shape"] == [4, 5] and clj_res[2]["data"][
                "shape"
            ] == [2, 5]:
                print("PASSED dataloader batch shapes")
            else:
                print(
                    f"FAILED dataloader batch shapes: {[b['data']['shape'] for b in clj_res]}"
                )
                all_passed = False
        else:
            print(f"FAILED dataloader batch count: {len(clj_res)}")
            all_passed = False
    else:
        all_passed = False

    # Test 3: Indexing (ix)
    print("\n--- Indexing (ix) ---")
    clj_res = run_clorch("""
        (let [t (torch/reshape (torch/tensor (map float (range 120))) [2 6 10])]
          {:select (torch/ix t 1)
           :slice  (torch/ix t :_ [1 4] :_)
           :ellipsis (torch/ix t 0 '... 1)
           :negative (torch/ix t -1 -1 -1)})
    """)
    if check_res(clj_res, "Indexing"):
        t = torch.arange(120).float().reshape(2, 6, 10)
        all_passed &= compare_tensors(clj_res["select"], t[1], "ix-selection")
        all_passed &= compare_tensors(clj_res["slice"], t[:, 1:4, :], "ix-slicing")
        all_passed &= compare_tensors(clj_res["ellipsis"], t[0, ..., 1], "ix-ellipsis")
        all_passed &= compare_tensors(clj_res["negative"], t[-1, -1, -1], "ix-negative")
    else:
        all_passed = False

    # Test 4: NN Modules
    print("\n--- NN Modules ---")
    clj_res = run_clorch("""
        (let [lin (nn/linear 5 2)
              input (torch/ones [1 5])]
          (nn/forward lin input))
    """)
    if check_res(clj_res, "NN Modules"):
        if clj_res["shape"] == [1, 2]:
            print("PASSED nn/linear forward shape")
        else:
            print(f"FAILED nn/linear forward shape: {clj_res['shape']}")
            all_passed = False
    else:
        all_passed = False

    # Test 5: Autograd
    print("\n--- Autograd ---")
    clj_res = run_clorch("""
        (let [x (torch/tensor [2.0] {:requires-grad true})
              y (torch/mul x x)]
          (autograd/backward y)
          (autograd/grad x))
    """)
    if check_res(clj_res, "Autograd"):
        x = torch.tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        all_passed &= compare_tensors(clj_res, x.grad, "autograd-x.grad")
    else:
        all_passed = False

    # Test 9: Comparison & Dtype conversion
    print("\n--- Comparison & Dtype ---")
    clj_res = run_clorch("""
        (let [t (torch/tensor [[1.0 5.0] [4.0 2.0]])]
          {:argmax-all (torch/argmax t)
           :argmax-dim (torch/argmax t 1)
           :eq-tensor  (torch/eq t (torch/tensor [[1.0 0.0] [4.0 0.0]]))
           :eq-scalar  (torch/eq t 5.0)
           :to-float   (torch/to-float (torch/tensor [1 2] {:dtype :int64}))
           :to-long    (torch/to-long (torch/tensor [1.1 2.9]))})
    """)
    if check_res(clj_res, "Comparison & Dtype"):
        t = torch.tensor([[1.0, 5.0], [4.0, 2.0]])
        all_passed &= compare_tensors(
            clj_res["argmax-all"], torch.argmax(t), "argmax-all"
        )
        all_passed &= compare_tensors(
            clj_res["argmax-dim"], torch.argmax(t, 1), "argmax-dim"
        )
        all_passed &= compare_tensors(
            clj_res["eq-tensor"],
            t.eq(torch.tensor([[1.0, 0.0], [4.0, 0.0]])),
            "eq-tensor",
        )
        all_passed &= compare_tensors(clj_res["eq-scalar"], t.eq(5.0), "eq-scalar")
        all_passed &= compare_tensors(
            clj_res["to-float"],
            torch.tensor([1, 2], dtype=torch.int64).float(),
            "to-float",
        )
        all_passed &= compare_tensors(
            clj_res["to-long"], torch.tensor([1.1, 2.9]).long(), "to-long"
        )
    else:
        all_passed = False

    # Test 10: Embedding Layer
    print("\n--- Embedding ---")

    # 10.1 Basic & from_pretrained
    # Pre-generate weights as list for exact parity
    w_list = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ]
    clj_res = run_clorch(f"""
        (let [w (torch/tensor {w_list})
              emb (nn/embedding-from-pretrained w :freeze false)
              idx (torch/tensor [[1 2] [4 5]] {{:dtype :int64}})]
          (nn/forward emb idx))
    """)
    if check_res(clj_res, "Embedding-basic"):
        py_w = torch.tensor(w_list)
        emb = torch.nn.Embedding.from_pretrained(py_w, freeze=False)
        idx = torch.tensor([[1, 2], [4, 5]], dtype=torch.long)
        all_passed &= compare_tensors(clj_res, emb(idx), "embedding-basic")

    # 10.2 padding_idx
    clj_res = run_clorch("""
        (let [emb (nn/embedding 10 3 :padding-idx 0)
              idx (torch/tensor [0] {:dtype :int64})]
          (nn/forward emb idx))
    """)
    if check_res(clj_res, "Embedding-padding"):
        if all(x == 0 for x in clj_res["data"]):
            print("PASSED embedding-padding (index 0 is zero)")
        else:
            print(f"FAILED embedding-padding: {clj_res['data']}")
            all_passed = False

    # 10.3 max_norm
    clj_res = run_clorch("""
        (let [w (torch/tensor [[1.0 1.0 1.0] [1.0 1.0 1.0]])
              emb (nn/embedding-from-pretrained w :max-norm 1.0 :freeze false)
              idx (torch/tensor [1] {:dtype :int64})]
          ;; After forward, weight should be renormalized
          (nn/forward emb idx)
          (get (nn/state-dict emb) :weight))
    """)
    if check_res(clj_res, "Embedding-max-norm"):
        # Norm of [1, 1, 1] is sqrt(3) ~ 1.732. Max norm 1.0 should scale it down to 1.0 total.
        # So each element becomes 1/sqrt(3) ~ 0.577
        val = clj_res["data"][3]  # index 1, first element
        if abs(val - 0.57735) < 1e-3:
            print("PASSED embedding-max-norm renormalization")
        else:
            print(f"FAILED embedding-max-norm: {val}")
            all_passed = False

    # Cleanup weight file
    import os

    if os.path.exists("emb_weights.pt"):
        os.remove("emb_weights.pt")

    if all_passed:
        print("\nALL COMPARISON TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nSOME COMPARISON TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
