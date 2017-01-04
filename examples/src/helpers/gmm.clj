(ns helpers.gmm
  (:require [clojure.core.matrix :as m]))

(defn normalize
  "Normalizes lst so that entries sum to one."
  [lst]
  (let [sum (reduce + lst)]
    (map #(/ % sum) lst)))

(defn move-to-unit-box [data]
  "Move data inside [-1, 1] box to a unit box. Ignores points outside of this
  box. Used with gridify."
  (remove nil? (map (fn [x]
                      (let [res (m/div (m/add 1 x) 2)
                            res (if (number? res)
                                  [res]
                                  res)]
                        (if (some #(or (> % 1) (< % 0)) res) nil res)))
                    data)))

(defn- walk-numbers [f lst]
  "Walks a nested list of numbers and applies function f to each number."
  (if (number? lst)
    (f lst)
    (map (partial walk-numbers f) lst)))

(defn- get-grid-index [x num-grids]
  "Takes in a point x from [0, 1] and outputs the correct grid index from
  {0, 1, ... (num-grids - 1)}."
  (let [res (int (m/floor (* x num-grids)))]
    (if (= res num-grids)
      (dec num-grids)
      res)))

(defn- zeros [dimensions]
  "Create a tensor of zeros of specified dimensions."
  (reduce #(vec (repeat %2 %1)) 0 (reverse dimensions)))

(defn grid-counts
  "Takes a list of data points (either numbers or vectors) inside a unit
  hypercube.
  Outputs a tensor of counts of points inside grids of a multivariate
  rectangular grid."
  ([data grid-dimensions]
   (loop [res (zeros grid-dimensions)
          grid-indices (mapv #(mapv get-grid-index % grid-dimensions)
                             data)]
     (if (empty? grid-indices)
       res
       (recur (update-in res (first grid-indices) inc)
              (rest grid-indices)))))
  ([data]
   (grid-counts data (repeat (count (first data)) 200))))

(defn gridify
  "Takes a list of data points (either numbers or vectors) inside a unit
  hypercube.
  Outputs a tensor of empirical probabilities of points inside grids of a
  multivariate rectangular grid."
  ([data grid-dimensions]
   (let [grid-counts (grid-counts data grid-dimensions)
         max-count (apply max (flatten grid-counts))]
     (walk-numbers (if (= max-count 0)
                     double
                     #(double (/ % max-count)))
                   grid-counts)))
  ([data]
   (gridify data (repeat (count (first data)) 200))))
