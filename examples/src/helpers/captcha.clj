(ns helpers.captcha
  (:require [clojure.core.matrix :as m]
            [anglican.runtime :refer :all]
            [helpers.hdf5 :refer [parse-datasets-hdf5]]))

(defdist random-projection-matrix
  "Returns the [k d] matrix R described in http://ftp10.us.freebsd.org/users/azhang/disc/PODS/pdf-files/27.pdf"
  [k d]
  [dist (categorical [[(sqrt (/ 3. k)) (/ 1 6)] [0 (/ 2. 3)] [(- (sqrt (/ 3. k))) (/ 1 6)]])]
  (sample* [this] (m/matrix (repeatedly k (fn [] (repeatedly d #(sample* dist))))))
  (observe* [this value] nil))

(defn gen-random-projection-matrix [source-dim target-dim]
  (sample* (random-projection-matrix target-dim source-dim)))

;; Dimensionality reduction for abc-dist
(def random-projection-matrices (parse-datasets-hdf5 "resources/random-projection-matrices.h5"))
(def dataset-name 'R10000-500) ; Possible datasets: 'R100, 'R200, 'R500, 'R1000, 'R2000. The number is the target dimension number.
(def R (m/matrix (get random-projection-matrices dataset-name)))
(defn reduce-dim [captcha]
  (m/mmul R (m/to-vector (apply concat captcha))))

(def levenshtein
  (memoize (fn [str1 str2]
             (let [len1 (count str1)
                   len2 (count str2)]
               (cond (zero? len1) len2
                     (zero? len2) len1
                     :else
                     (let [cost (if (= (first str1) (first str2)) 0 1)]
                       (min (inc (levenshtein (rest str1) str2))
                            (inc (levenshtein str1 (rest str2)))
                            (+ cost
                               (levenshtein (rest str1) (rest str2))))))))))
(defn levenshtein-normalized [str1 str2]
  (let [len1 (count str1)
        len2 (count str2)]
    (float (/ (levenshtein str1 str2) (max len1 len2)))))
