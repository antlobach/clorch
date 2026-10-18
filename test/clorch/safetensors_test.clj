(ns clorch.safetensors-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.safetensors :as safe]
            [clorch.torch :as t])
  (:import [java.io FileOutputStream]
           [java.nio.file Files Path]
           [org.bytedeco.pytorch Tensor]))

(set! *warn-on-reflection* true)

(defn- temporary-file
  ^Path []
  (Files/createTempFile "clorch-safetensors-test-" ".safetensors"
                        (make-array java.nio.file.attribute.FileAttribute 0)))

(defn- max-error
  [^Tensor actual ^Tensor expected]
  (t/item-float
   (t/max
    (t/abs
     (t/sub (t/to actual :float32)
            (t/to expected :float32))))))

(deftest small-round-trip-preserves-values-dtypes-and-metadata
  (let [file (temporary-file)]
    (try
      (t/with-torch
        (let [values [1.25 -2.5 3.0]
              tensors {"bool" (t/tensor [true false true] {:dtype :bool})
                       "bfloat16" (t/tensor values {:dtype :bfloat16})
                       "float16" (t/tensor values {:dtype :float16})
                       "float32" (t/tensor values)
                       "float64" (t/tensor values {:dtype :float64})
                       "int8" (t/tensor [1 -2 3] {:dtype :int8})
                       "int32" (t/tensor [1 -2 3] {:dtype :int32})
                       "int64" (t/tensor [1 -2 3] {:dtype :int64})
                       "uint8" (t/tensor [1 2 3] {:dtype :uint8})}]
          (safe/save-file! tensors file :metadata {"source" "small-test"})
          (let [loaded (safe/load-file file)]
            (is (= (set (keys tensors)) (set (keys loaded))))
            (is (= {"source" "small-test"} (safe/metadata file)))
            (doseq [[tensor-name expected] tensors]
              (testing tensor-name
                (is (= (.toString (.scalar_type ^Tensor expected))
                       (.toString (.scalar_type ^Tensor (get loaded tensor-name)))))
                (is (zero? (max-error (get loaded tensor-name) expected))))))))
      (finally
        (Files/deleteIfExists file)))))

(deftest malformed-data-offset-is-rejected
  (let [file (temporary-file)
        ^bytes header-bytes
        (.getBytes
         "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}")
        padding (mod (- 8 (mod (alength header-bytes) 8)) 8)
        padded (byte-array (+ (alength header-bytes) padding))
        contents (java.nio.ByteBuffer/allocate (+ 8 (alength padded) 4))]
    (try
      (System/arraycopy header-bytes 0 padded 0 (alength header-bytes))
      (java.util.Arrays/fill padded (alength header-bytes) (alength padded) (byte 32))
      (.order contents java.nio.ByteOrder/LITTLE_ENDIAN)
      (.putLong contents (long (alength padded)))
      (.put contents padded)
      (.putFloat contents 1.0)
      (with-open [output (FileOutputStream. (.toFile file))]
        (.write output ^bytes (.array contents)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Invalid SafeTensors tensor entry"
                            (safe/load-file file)))
      (finally
        (Files/deleteIfExists file)))))
