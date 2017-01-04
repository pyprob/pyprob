(ns helpers.general)

(defn- min-index [v]
  (let [length (count v)]
    (loop [minimum (first v)
           min-index 0
           i 1]
      (if (< i length)
        (let [value (nth v i)]
          (if (< value minimum)
            (recur value i (inc i))
            (recur minimum min-index (inc i))))
        min-index))))

(defn- max-index [v]
  (min-index (mapv - v)))

(defn empirical-MAP
  "calculates a maximum a-posteriori value from weighted samples;
  - accepts a map or sequence of log weighted
  values [x log-w].
  - returns the x with largest log-w"
  [weighted]
  (nth (map first weighted) (max-index (map second weighted))))
