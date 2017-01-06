(ns queries.minimal
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery]]))

(defquery minimal [obs]
  (let [b (sample "b" (beta 1 1))
        disc (sample "disc" (discrete [1 1 1 1 1]))
        flp (sample "flp" (flip 0.5))
        g (sample "g" (gamma 1 2))
        mvnn (sample "mvnn" (mvn [0 0] [[1 0] [0 1]]))
        n (sample "n" (normal 0 1))
        p (sample "p" (poisson 5))
        u-c (sample "u-c" (uniform-continuous 2.2 5.5))
        u-d (sample "u-d" (uniform-discrete 2 5))]
    (observe (mvn [0 0] [[1 0] [0 1]]) obs)
    b))

(defn COMPILE-combine-observes-fn [observes]
  (:value (first observes)))
