(ns clorch.distributed.launch
  "Local multi-process launcher used by REPLs and production entrypoints."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import [java.io File RandomAccessFile]
           [java.net ServerSocket]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files]
           [java.time Instant]
           [java.util UUID]
           [java.util.concurrent LinkedBlockingQueue TimeUnit TimeoutException]))

(declare stop!)

(defrecord WorkerProcess [rank ^Process process ^File stdout ^File stderr])

(defrecord DistributedJob
           [id config workers status started-at monitor config-file]
  java.lang.AutoCloseable
  (close [this]
    (stop! this)))

(defn- ensure-java-25! []
  (let [feature (.feature (Runtime/version))]
    (when (< feature 25)
      (throw (ex-info "Distributed workers require Java 25 or newer"
                      {:java-feature feature})))))

(defn- available-port []
  (with-open [socket (ServerSocket. 0)]
    (.setReuseAddress socket true)
    (.getLocalPort socket)))

(defn- normalize-config [config]
  (let [nproc (long (get config :nproc-per-node 1))
        backend (keyword (get config :backend :nccl))
        main (:main config)
        devices (:devices config)
        timeout-ms (long (get config :timeout-ms 300000))]
    (when-not (pos? nproc)
      (throw (ex-info ":nproc-per-node must be positive"
                      {:nproc-per-node nproc})))
    (when-not (= :nccl backend)
      (throw (ex-info ":backend must be :nccl"
                      {:backend backend})))
    (when-not (qualified-symbol? main)
      (throw (ex-info ":main must be a namespace-qualified symbol"
                      {:main main})))
    (when-not (pos? timeout-ms)
      (throw (ex-info ":timeout-ms must be positive"
                      {:timeout-ms timeout-ms})))
    (when (and devices (not= nproc (count devices)))
      (throw (ex-info "Device count must equal :nproc-per-node"
                      {:nproc-per-node nproc :devices devices})))
    (when (and devices (not= (count devices) (count (set devices))))
      (throw (ex-info "Each distributed worker requires a distinct CUDA device"
                      {:devices devices})))
    (when (nil? devices)
      (throw (ex-info "NCCL launch requires explicit :devices"
                      {:nproc-per-node nproc})))
    (assoc config
           :nproc-per-node nproc
           :backend backend
           :timeout-ms timeout-ms
           :master-address (str (get config :master-address "127.0.0.1"))
           :master-port (long (get config :master-port (available-port)))
           :args (get config :args {}))))

(defn- ensure-edn! [value]
  (let [encoded (pr-str value)
        decoded (edn/read-string encoded)]
    (when-not (= value decoded)
      (throw (ex-info "Worker configuration must round-trip through EDN"
                      {:value value})))
    encoded))

(defn- create-log-directory [config job-id]
  (let [directory (or (:log-dir config)
                      (str (System/getProperty "java.io.tmpdir")
                           File/separator "clorch-distributed-" job-id))
        path (.toPath (io/file directory))]
    (Files/createDirectories path (make-array java.nio.file.attribute.FileAttribute 0))
    (.toFile path)))

(defn- write-worker-config! [config log-directory]
  (let [path (.toPath (io/file log-directory "worker-config.edn"))
        content (ensure-edn! (select-keys config [:main :args]))]
    (Files/writeString path content StandardCharsets/UTF_8
                       (into-array java.nio.file.OpenOption []))
    (.toFile path)))

(defn- java-command [config config-file]
  (let [java-home (or (:java-home config) (System/getProperty "java.home"))
        java-executable (str java-home File/separator "bin" File/separator "java")
        classpath (System/getProperty "java.class.path")]
    (vec (concat [java-executable]
                 (:jvm-opts config)
                 ["-cp" classpath
                  "clojure.main" "-m" "clorch.distributed.worker"
                  (.getAbsolutePath ^File config-file)]))))

(defn- configure-environment!
  [^ProcessBuilder builder config rank]
  (let [environment (.environment builder)
        devices (:devices config)
        assigned-device (when devices (nth devices rank))
        cuda-local-rank (if assigned-device 0 rank)]
    (doseq [[environment-key value] (:environment config)]
      (.put environment (name environment-key) (str value)))
    (doseq [[environment-key value] {"RANK" rank
                                     "LOCAL_RANK" cuda-local-rank
                                     "WORLD_SIZE" (:nproc-per-node config)
                                     "LOCAL_WORLD_SIZE" (:nproc-per-node config)
                                     "MASTER_ADDR" (:master-address config)
                                     "MASTER_PORT" (:master-port config)
                                     "CLORCH_DIST_BACKEND" (name (:backend config))
                                     "CLORCH_DIST_TIMEOUT_MS" (:timeout-ms config)}]
      (.put environment environment-key (str value)))
    (when assigned-device
      (.put environment "CUDA_VISIBLE_DEVICES" (str assigned-device)))
    builder))

(defn- start-worker! [config config-file log-directory rank]
  (let [stdout (io/file log-directory (str "rank-" rank ".out.log"))
        stderr (io/file log-directory (str "rank-" rank ".err.log"))
        builder (ProcessBuilder. ^java.util.List (java-command config config-file))]
    (.directory builder (io/file (or (:working-directory config)
                                     (System/getProperty "user.dir"))))
    (.redirectOutput builder stdout)
    (.redirectError builder stderr)
    (configure-environment! builder config rank)
    (->WorkerProcess rank (.start builder) stdout stderr)))

(defn- terminate-process! [^Process process]
  (when (.isAlive process)
    (.destroy process)
    (when-not (.waitFor process 10 TimeUnit/SECONDS)
      (.destroyForcibly process)
      (.waitFor process 10 TimeUnit/SECONDS))))

(defn- monitor-workers [workers status config-file]
  (let [completions (LinkedBlockingQueue.)]
    (doseq [{:keys [rank ^Process process]} workers]
      (future
        (.put completions [rank (.waitFor process)])))
    (loop [remaining (count workers)
           exits (sorted-map)
           failure nil]
      (if (zero? remaining)
        (let [final-status (if failure
                             {:state :failed
                              :failed-rank (first failure)
                              :exit-code (second failure)
                              :exit-codes exits
                              :finished-at (Instant/now)}
                             {:state :succeeded
                              :exit-codes exits
                              :finished-at (Instant/now)})]
          (reset! status final-status)
          (Files/deleteIfExists (.toPath ^File config-file))
          final-status)
        (let [[rank exit-code] (.take completions)
              first-failure (or failure (when-not (zero? exit-code)
                                          [rank exit-code]))]
          (when (and (nil? failure) first-failure)
            (doseq [{:keys [^Process process]} workers]
              (terminate-process! process)))
          (recur (dec remaining)
                 (assoc exits rank exit-code)
                 first-failure))))))

(defn launch!
  "Launches one local JVM per distributed rank and returns a job handle.

  Required: :main qualified worker function. NCCL also requires :devices, one
  CUDA device identifier per rank. Worker stdout and stderr are file-backed and
  never accumulated in coordinator memory."
  [config]
  (ensure-java-25!)
  (let [config (normalize-config config)
        job-id (str (UUID/randomUUID))
        log-directory (create-log-directory config job-id)
        config-file (write-worker-config! config log-directory)
        status (atom {:state :starting})
        workers (atom [])]
    (try
      (dotimes [rank (:nproc-per-node config)]
        (swap! workers conj (start-worker! config config-file log-directory rank)))
      (reset! status {:state :running})
      (let [monitor (future (monitor-workers @workers status config-file))]
        (->DistributedJob job-id config @workers status (Instant/now)
                          monitor config-file))
      (catch Throwable cause
        (doseq [{:keys [^Process process]} @workers]
          (terminate-process! process))
        (Files/deleteIfExists (.toPath config-file))
        (throw cause)))))

(defn status
  "Returns immutable job and per-rank process status."
  [job]
  {:id (:id job)
   :state @(:status job)
   :started-at (:started-at job)
   :workers (mapv (fn [{:keys [rank ^Process process stdout stderr]}]
                    {:rank rank
                     :pid (.pid process)
                     :alive? (.isAlive process)
                     :stdout (.getAbsolutePath ^File stdout)
                     :stderr (.getAbsolutePath ^File stderr)})
                  (:workers job))})

(defn await!
  "Waits for every rank. Throws when a rank fails or timeout elapses."
  ([job]
   (await! job nil))
  ([job timeout-ms]
   (let [result (if timeout-ms
                  (deref (:monitor job) timeout-ms ::timeout)
                  @(:monitor job))]
     (when (= ::timeout result)
       (throw (TimeoutException.
               (str "Distributed job exceeded " timeout-ms " ms"))))
     (when (= :failed (:state result))
       (throw (ex-info "Distributed job failed" result)))
     result)))

(defn stop!
  "Terminates all live ranks and waits for their process trees to exit."
  [job]
  (doseq [{:keys [^Process process]} (:workers job)]
    (terminate-process! process))
  (when-not (realized? (:monitor job))
    (reset! (:status job) {:state :stopping}))
  @(:monitor job))

(defn log-paths
  "Returns rank-indexed stdout and stderr paths."
  [job]
  (into (sorted-map)
        (map (fn [{:keys [rank stdout stderr]}]
               [rank {:stdout (.getAbsolutePath ^File stdout)
                      :stderr (.getAbsolutePath ^File stderr)}])
             (:workers job))))

(defn- read-last-bytes [^File file max-bytes]
  (if-not (.exists file)
    ""
    (with-open [input (RandomAccessFile. file "r")]
      (let [length (.length input)
            start (max 0 (- length max-bytes))
            size (int (- length start))
            payload (byte-array size)]
        (.seek input start)
        (.readFully input payload)
        (String. payload StandardCharsets/UTF_8)))))

(defn logs
  "Returns a bounded tail of one rank's stdout and stderr."
  ([job rank]
   (logs job rank 65536))
  ([job rank max-bytes]
   (when-not (pos? max-bytes)
     (throw (ex-info "Log byte limit must be positive"
                     {:max-bytes max-bytes})))
   (let [{:keys [stdout stderr]}
         (or (some #(when (= rank (:rank %)) %) (:workers job))
             (throw (ex-info "Unknown distributed rank" {:rank rank})))]
     {:rank rank
      :stdout (read-last-bytes stdout max-bytes)
      :stderr (read-last-bytes stderr max-bytes)})))
