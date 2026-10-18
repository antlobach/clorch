# SafeTensors I/O

The `clorch.safetensors` namespace reads and writes [SafeTensors](https://huggingface.co/docs/safetensors) archives without Python, pickle execution, or Hugging Face tooling. Every dtype in the SafeTensors specification is supported.

## Reading

```clojure
(require '[clorch.safetensors :as safe])

(def tensors (safe/load-file "model.safetensors"))
;; => {"layer.weight" #object[Tensor ...], ...}
```

- Returns a map of string tensor names to CPU tensors with the shapes and dtypes stored in the file.
- Tensor data is validated before reading: dtype, shape, byte counts, and non-overlapping offsets covering the whole payload.
- The returned tensors are caller-owned; release them with their enclosing native-memory scope (see [Memory Management](memory.md)).
- Pickle payloads are never executed; only the JSON header and raw tensor bytes are read.

```clojure
(safe/metadata "model.safetensors")
;; => {"format" "pt"} — the file's __metadata__ map, {} when absent
```

## Writing

```clojure
(safe/save-file! tensors "out.safetensors"
                 :metadata {"format" "pt" "source" "clorch"})
```

- Tensors are detached, made contiguous, and copied to CPU before their bytes are written.
- Optional `:metadata` must map strings to strings.
- The file is written to a temporary file in the destination directory and moved into place atomically, so readers never observe a partial archive.

## Loading Model Weights

`load-state-dict!` loads an archive whose tensor names are the model's flat dotted parameter paths (see `nn/flat-parameter-names`):

```clojure
(nn/flat-parameter-names model)
;; => #{"l1.weight" "l1.bias" "l2.weight" "l2.bias"}

(safe/load-state-dict! model "model.safetensors")
```

Every supplied name is copied into the matching parameter; missing or unexpected names raise an error instead of being skipped. To adapt foreign checkpoint names to your model's names, load with `safe/load-file`, transform the map, and call `nn/load-flat-state-dict!` directly.
