(ns queries.gmm-variable-number-of-clusters
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery with-primitive-procedures]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as mlin]
            [helpers.gmm :refer [normalize move-to-unit-box gridify]]))

;; These are the hyperparameters for the model:
;; mu0		mean for the Gaussian prior over the cluster means
;; Sigma0	covariance for the Gaussian prior over the cluster means
(def hyperparameters {:mu-0 [0 0]
                      :Sigma-0 [[0.1 0] [0 0.1]]})

;; This is a hack in order to sample from a discrete distribution in such a way that it doesn't trigger adaptation of the proposal. This will not change the inference behaviour in particle-based algorithms since it is equivalent to having a bootstrap proposal
(defn sample-discrete [probs]
  (sample* (discrete probs)))
(defn sort-means-vars [means-vars]
  (sort-by (comp mlin/norm first) means-vars))
(def lo 0.0008)
(def hi 0.0025)

(with-primitive-procedures [normalize m/mmul m/identity-matrix sample-discrete sort-means-vars]
  (defquery gmm-variable-number-of-clusters [data hyperparameters]
    (let [num-clusters (sample "num-clusters" (uniform-discrete 1 6))
          cluster-probs (normalize (repeat num-clusters 1))
          means-vars (repeatedly num-clusters #(list (sample "mean" (mvn (:mu-0 hyperparameters) (:Sigma-0 hyperparameters)))
                                                     (+ lo (* (/ (sample "var" (uniform-discrete 0 21)) 20.0)
                                                              (- hi lo)))))
          sorted-means-vars (sort-means-vars means-vars)
          means (map first sorted-means-vars)
          vars (map second sorted-means-vars)
          clusters (map #(let [cluster (sample-discrete cluster-probs)
                               mean (nth means cluster)
                               var (nth vars cluster)]
                           (observe (mvn mean (mmul var (identity-matrix 2))) %)
                           cluster)
                        data)]
      {:num-clusters num-clusters
       :clusters clusters
       :cluster-probs cluster-probs
       :means means
       :vars vars})))

(def num-data-points 100)
(def grid-dimensions [100 100])

;; Takes in samples from the prior, sorts means and vars according to means' norm
(defn combine-samples-fn [samples]
  (let [mean-samples (filter #(= "mean" (:sample-address %)) samples)
        var-samples (filter #(= "var" (:sample-address %)) samples)
        var-mean-zip (map vector var-samples mean-samples)
        sorted-var-mean-zip (sort-by (comp mlin/norm :value second) var-mean-zip)
        sorted-var-samples (map first sorted-var-mean-zip)
        sorted-mean-samples (map second sorted-var-mean-zip)
        sorted-samples (loop [res []
                              smps samples
                              var-smps sorted-var-samples
                              mean-smps sorted-mean-samples]
                         (if (empty? smps)
                           res
                           (let [smp (first smps)]
                             (case (:sample-address smp)
                               "mean" (recur (conj res (assoc smp :value (:value (first mean-smps))))
                                             (rest smps)
                                             var-smps
                                             (rest mean-smps))
                               "var" (recur (conj res (assoc smp :value (:value (first var-smps))))
                                            (rest smps)
                                            (rest var-smps)
                                            mean-smps)
                               (recur (conj res smp)
                                      (rest smps)
                                      var-smps
                                      mean-smps)))))]
    sorted-samples))

;; Takes in observes from the prior, puts the on the grid.
(defn combine-observes-fn [observes]
  (gridify (move-to-unit-box (mapv (comp vec :value) observes)) grid-dimensions))
