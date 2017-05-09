;; gorilla-repl.fileformat = 1

;; **
;;; # Uniform Continuous
;; **

;; **
;;; Import the minimum required libraries to run define queries, compile them, and run the inference using compiled artifacts.
;; **

;; @@
(ns worksheets.uniform-continuous
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery]]
            [anglican.stat :refer [empirical-distribution collect-predicts collect-by]]
            anglican.infcomp.csis
            anglican.importance
            [anglican.infcomp.dists :refer :all]
            [anglican.infcomp.network :refer :all]
            [anglican.inference :refer [infer]])
  (:use [gorilla-plot core]))
;; @@

;; **
;;; Define a query
;; **

;; @@
(defquery uniform-continuous-query [obs]
  (let [uc-min (sample "uc-min" (normal 0 0.1))
        uc-range (sample "uc-range" (normal 10 0.1))
        uc-max (+ uc-min uc-range)
        mean (sample "mean" (uniform-continuous uc-min uc-max))
        std 0.01]
    (observe (normal mean std) obs)
    {:mean mean
     :uc-min uc-min
     :uc-max uc-max}))
;; @@

;; **
;;; ## Inference Compilation
;; **

;; **
;;; Specify a function `combine-observes-fn` that combines observes from a sample in a form suitable for Torch:
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@

;; **
;;; In order to write this function, you'll need to look at how a typical `observes` object from your query looks like:
;; **

;; @@
(sample-observes-from-prior uniform-continuous-query [1])
;; @@

;; **
;;; Compile our query
;; **

;; @@
(def torch-connection (start-torch-connection uniform-continuous-query [1] combine-observes-fn))
;; @@

;; **
;;; Then run the following to train the neural network:
;;; 
;;; `python -m infcomp.compile`
;; **

;; @@
(stop-torch-connection torch-connection)
;; @@

;; **
;;; ## Inference
;; **

;; **
;;; First, run the inference server from torch-infcomp:
;;; 
;;; `python -m infcomp.infer`
;;; 
;;; Then run the query, specifying number of particles:
;; **

;; @@
(def num-particles 100)
(def csis-states (take num-particles (infer :csis uniform-continuous-query [5] :tcp-endpoint "tcp://pindarus:6666")))
(list-plot (vec (empirical-distribution (collect-by (comp :mean :result) csis-states))) :y-title "mean")
(list-plot (vec (empirical-distribution (collect-by (comp :uc-min :result) csis-states))) :y-title "uc-min")
(list-plot (vec (empirical-distribution (collect-by (comp :uc-max :result) csis-states))) :y-title "uc-max")
;; @@

;; **
;;; Using the same number of particles, we don't have such good results with importance sampling:
;; **

;; @@
(def num-particles 100)
(def is-states (take num-particles (infer :importance uniform-continuous-query [5])))
(list-plot (vec (empirical-distribution (collect-by (comp :mean :result) is-states))) :y-title "mean")
(list-plot (vec (empirical-distribution (collect-by (comp :uc-min :result) is-states))) :y-title "uc-min")
(list-plot (vec (empirical-distribution (collect-by (comp :uc-max :result) is-states))) :y-title "uc-max")
;; @@
