(ns autograd-tutorial
  "Port of Sebastian Raschka's 'Automatic Differentiation Made Easy'
   Source: https://sebastianraschka.com/teaching/pytorch-1h/#4-automatic-differentiation-made-easy"
  (:require [clorch.torch :as torch]
            [clorch.autograd :as autograd]))

(println "--- Section 4: Automatic Differentiation Made Easy ---")

(torch/with-torch
  (let [x (torch/tensor 2.0 {:requires-grad true})
        y (torch/mul x x)]

    (println "\n1. Basic Autograd: y = x^2 at x=2")
    (println "x:")
    (torch/tprint x)
    (println "y:")
    (torch/tprint y)

    (autograd/backward y)

    (let [g (autograd/grad x)]
      (println "dy/dx (should be 2x = 4):")
      (torch/tprint g)))

  (println "\n2. Sequence iteration test (tseq):")
  (let [t (torch/reshape (torch/tensor (map float (range 6))) [3 2])]
    (doseq [slice (torch/tseq t)]
      (println "Slice shape:" (torch/size slice) "Data:" (torch/item-float (torch/ix slice 0))))))

(println "\n--- Tutorial Complete ---")

