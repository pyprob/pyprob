(ns queries.gaussian
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery]]))

(defquery gaussian [obs]
  (let [mean (sample "mean" (normal 0 1))
        std 0.1]
    (observe (normal mean std) obs)
    mean))
