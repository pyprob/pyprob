;; gorilla-repl.fileformat = 1

;; **
;;; # Minimal example
;; **

;; **
;;; Import the minimum required libraries to run define queries, compile them, and run the inference using compiled artifacts.
;; **

;; @@
(ns minimal
  (:require anglican.infcomp.csis
            [anglican.infcomp.network :refer :all]
            [anglican.inference :refer [infer]])
  (:use [anglican emit runtime]
        [gorilla-plot core]))
;; @@

;; **
;;; Define a (non-sensical) query:
;; **

;; @@
(defquery q [observes]
  (let [b (sample "b" (beta 1 1))
        disc (sample "disc" (discrete [1 1 1 1 1]))
        flp (sample "flp" (flip 0.5))
        g (sample "g" (gamma 1 2))
        mvnn (sample "mvnn" (mvn [0 0] [[1 0] [0 1]]))
        n (sample "n" (normal 0 1))
        p (sample "p" (poisson 5))
        u-c (sample "u-c" (uniform-continuous 2.2 5.5))
        u-d (sample "u-d" (uniform-discrete 2 5))]
    (observe (mvn [0 0] [[1 0] [0 1]]) observes)
    b))
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
(sample-observes-from-prior q [[1 2]])
;; @@

;; **
;;; Now, we can compile our query. First, establish a connection with the Torch side:
;; **

;; @@
(def torch-connection (start-torch-connection q [[1 2]] combine-observes-fn))
;; @@

;; **
;;; Now, run `compile.lua` from the [Torch side](https://github.com/tuananhle7/torch-csis) until you're happy, e.g.:
;;;
;;; `th compile.lua --batchSize 5 --validSize 5 --validInterval 100 --obsEmbDim 5 --lstmDim 5`
;;;
;;; When you're happy, you can stop the training by running the following (also, end Torch training via Ctrl-C):
;; **

;; @@
(stop-torch-connection torch-connection)
;; @@

;; **
;;; ## Inference
;; **

;; **
;;; To run inference using the compiled artifact, first run `infer.lua` on the [Torch side](https://github.com/tuananhle7/torch-csis) to start the inference server, e.g.:
;;;
;;; `th infer.lua --latest`
;;;
;;; Then run the query, specifying number of particles:
;; **

;; @@
(def num-particles 10)
(def csis-states (take num-particles (infer :csis q [[1 2]])))
;; @@

;; **
;;; Look at the 10th particle:
;; **

;; @@
(last csis-states)
;; @@
