(ns helpers.hdf5
  (:require [clojure.core.matrix :as m]
            [clj-hdf5.core :as hdf5]
            [clojure.string :as str]))

(defn- array-to-mat-vec [array]
  "Takes in either 1D or 2D Java array and returns either Clojure vector or matrix."
  (let [temp (seq array)]
    (if (number? (first temp))
      (m/to-vector temp)
      (m/matrix (map seq temp)))))

(defn parse-datasets-hdf5 [filename]
  "Takes filename of HDF5 file containing only 1D and 2D datasets,

  Returns hashmap of the same structure as the group structure with
  hashmap keys being group names."
  (let [h5root (hdf5/open (clojure.java.io/file filename))
        datasets-and-paths (remove nil? (doall (hdf5/walk h5root #(if (hdf5/dataset? %)
                                                               {:dataset (hdf5/read %) :path (:path %)}
                                                               nil))))
        result (reduce (fn [nn-params dataset-and-path]
                         (let [dataset (array-to-mat-vec (:dataset dataset-and-path))
                               path (map read-string (remove #(= % "")
                                                             (str/split (:path dataset-and-path) #"/")))]
                           (assoc-in nn-params path dataset)))
                       {}
                       datasets-and-paths)]
    (hdf5/close h5root)
    result))

;; Code for pre-generating random projection matrices
;; (def WIDTH 200)
;; (def HEIGHT 50)
;; (def R100 (sample (random-projection-matrix 100 (* WIDTH HEIGHT))))
;; (def R200 (sample (random-projection-matrix 200 (* WIDTH HEIGHT))))
;; (def R500 (sample (random-projection-matrix 500 (* WIDTH HEIGHT))))
;; (def R1000 (sample (random-projection-matrix 1000 (* WIDTH HEIGHT))))
;; (def R2000 (sample (random-projection-matrix 2000 (* WIDTH HEIGHT))))
;; (def writer (HDF5Factory/open "resources/random-projection-matrices.h5"))
;; (.writeDoubleMatrix writer "/R100" (mat-vec-to-array (vec (map vec R100))))
;; (.writeDoubleMatrix writer "/R200" (mat-vec-to-array (vec (map vec R200))))
;; (.writeDoubleMatrix writer "/R500" (mat-vec-to-array (vec (map vec R500))))
;; (.writeDoubleMatrix writer "/R1000" (mat-vec-to-array (vec (map vec R1000))))
;; (.writeDoubleMatrix writer "/R2000" (mat-vec-to-array (vec (map vec R2000))))
;; (.delete writer "__DATA_TYPES__")
;; (.close writer)
