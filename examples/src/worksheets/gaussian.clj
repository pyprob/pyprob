;; gorilla-repl.fileformat = 1

;; **
;;; # Gaussian
;; **

;; **
;;; Import the minimum required libraries to run define queries, compile them, and run the inference using compiled artifacts.
;; **

;; @@
(ns worksheets.gaussian
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery]]
            [anglican.stat :refer [empirical-distribution collect-results]]
            anglican.infcomp.csis
            anglican.importance
            [anglican.infcomp.network :refer :all]
            [anglican.inference :refer [infer]])
  (:use [gorilla-plot core]))
;; @@

;; **
;;; Define a Gaussian unknown mean model
;; **

;; @@
(defquery gaussian [obs]
  (let [mean (sample "mean" (normal 0 1))
        std 0.1]
    (observe (normal mean std) obs)
    mean))
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
(sample-observes-from-prior gaussian [1])
;; @@

;; **
;;; Compile our query
;; **

;; @@
(def torch-connection (start-torch-connection gaussian [1] combine-observes-fn))
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
(def csis-states (take num-particles (infer :csis gaussian [2.3])))
(list-plot (vec (empirical-distribution (collect-results csis-states))))
;; @@

;; **
;;; Using the same number of particles, we don't have such good results with importance sampling:
;; **

;; @@
(def num-particles 100)
(def is-states (take num-particles (infer :importance gaussian [2.3])))
(list-plot (vec (empirical-distribution (collect-results is-states))))
;; @@
