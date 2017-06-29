;; gorilla-repl.fileformat = 1

;; **
;;; # Uniform Continuous Ranges
;; **

;; **
;;; Import the minimum required libraries to run define queries, compile them, and run the inference using compiled artifacts.
;; **

;; @@
(ns worksheets.uniform-continuous-ranges
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer :all]
            [anglican.stat :as stat]
            [anglican.infcomp.zmq :as zmq]
            [anglican.inference :refer [infer]]
            [anglican.infcomp.prior :as prior]
            [gorilla-plot.core :as plt]
            anglican.infcomp.csis
            anglican.importance
            anglican.infcomp.core))

(anglican.infcomp.core/reset-infcomp-addressing-scheme!)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--26877$fn--26878]</span>","value":"#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--26877$fn--26878]"}
;; <=

;; **
;;; ## Model
;; **

;; @@
(defquery uniform-continuous-ranges [obs]
  (let [uc-min (sample (categorical [[0 1] [100 1] [200 1]]))
        uc-range (sample (normal 10 0.1))
        uc-max (+ uc-min uc-range)
        mean (sample (uniform-continuous uc-min uc-max))
        std 0.1]
    (observe (normal mean std) obs)
    {:mean mean
     :uc-min uc-min
     :uc-max uc-max}))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous-ranges/uniform-continuous-ranges</span>","value":"#'worksheets.uniform-continuous-ranges/uniform-continuous-ranges"}
;; <=

;; **
;;; ## Inference Compilation
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous-ranges/combine-observes-fn</span>","value":"#'worksheets.uniform-continuous-ranges/combine-observes-fn"}
;; <=

;; @@
(prior/sample-observes-from-prior uniform-continuous-ranges [1])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O4</span>","value":"O4"}],"value":"[:observe-address O4]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-double'>203.94875531797743</span>","value":"203.94875531797743"}],"value":"[:value 203.94875531797743]"}],"value":"{:time-index 0, :observe-address O4, :observe-instance 0, :value 203.94875531797743}"}],"value":"[{:time-index 0, :observe-address O4, :observe-instance 0, :value 203.94875531797743}]"}
;; <=

;; @@
(def replier (zmq/start-replier uniform-continuous-ranges [1] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous-ranges/replier</span>","value":"#'worksheets.uniform-continuous-ranges/replier"}
;; <=

;; @@
(zmq/stop-replier replier)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;ZMQ connection terminated.&quot;</span>","value":"\"ZMQ connection terminated.\""}
;; <=

;; **
;;; ## Inference
;; **

;; @@
(def num-particles 100)
(def csis-states (take num-particles (infer :csis uniform-continuous-ranges [9])))
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :mean :result) csis-states)))))) :normalize :probability-density :y-title "mean")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-min :result) csis-states)))))) :normalize :probability-density :y-title "uc-min")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-max :result) csis-states)))))) :normalize :probability-density :y-title "uc-max")
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"33853df5-4ae3-4338-b7ec-99a5df557cc6","values":[{"x":10.025055600262325,"y":0},{"x":11.025055600262325,"y":1.0},{"x":12.025055600262325,"y":0}]}],"marks":[{"type":"line","from":{"data":"33853df5-4ae3-4338-b7ec-99a5df557cc6"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"y":{"scale":"y","field":"data.y"},"interpolate":{"value":"step-before"},"fill":{"value":"steelblue"},"fillOpacity":{"value":0.4},"stroke":{"value":"steelblue"},"strokeWidth":{"value":2},"strokeOpacity":{"value":1}}}}],"scales":[{"name":"x","type":"linear","range":"width","zero":false,"domain":{"data":"33853df5-4ae3-4338-b7ec-99a5df557cc6","field":"data.x"}},{"name":"y","type":"linear","range":"height","nice":true,"zero":false,"domain":{"data":"33853df5-4ae3-4338-b7ec-99a5df557cc6","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y","title":"uc-max","titleOffset":45}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"33853df5-4ae3-4338-b7ec-99a5df557cc6\", :values ({:x 10.025055600262325, :y 0} {:x 11.025055600262325, :y 1.0} {:x 12.025055600262325, :y 0})}], :marks [{:type \"line\", :from {:data \"33853df5-4ae3-4338-b7ec-99a5df557cc6\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :y {:scale \"y\", :field \"data.y\"}, :interpolate {:value \"step-before\"}, :fill {:value \"steelblue\"}, :fillOpacity {:value 0.4}, :stroke {:value \"steelblue\"}, :strokeWidth {:value 2}, :strokeOpacity {:value 1}}}}], :scales [{:name \"x\", :type \"linear\", :range \"width\", :zero false, :domain {:data \"33853df5-4ae3-4338-b7ec-99a5df557cc6\", :field \"data.x\"}} {:name \"y\", :type \"linear\", :range \"height\", :nice true, :zero false, :domain {:data \"33853df5-4ae3-4338-b7ec-99a5df557cc6\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\", :title \"uc-max\", :titleOffset 45}]}}"}
;; <=

;; @@
(def num-particles 100)
(def is-states (take num-particles (infer :importance uniform-continuous-ranges [5])))
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :mean :result) is-states)))))) :normalize :probability-density :y-title "mean")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-min :result) is-states)))))) :normalize :probability-density :y-title "uc-min")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-max :result) is-states)))))) :normalize :probability-density :y-title "uc-max")
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"fd662279-81b0-49bc-8f2a-57f78e64608b","values":[{"x":9.995690640909293,"y":0},{"x":10.006700867248437,"y":89.33512987858293},{"x":10.01771109358758,"y":0.0},{"x":10.028721319926724,"y":0.0},{"x":10.039731546265868,"y":0.0},{"x":10.050741772605011,"y":0.0},{"x":10.061751998944155,"y":0.0},{"x":10.072762225283299,"y":0.0},{"x":10.083772451622442,"y":0.0},{"x":10.094782677961586,"y":0.0},{"x":10.10579290430073,"y":0.0},{"x":10.116803130639873,"y":0.0},{"x":10.127813356979017,"y":0.0},{"x":10.13882358331816,"y":0.0},{"x":10.149833809657304,"y":0.0},{"x":10.160844035996448,"y":0.0},{"x":10.171854262335591,"y":1.4895243290044329},{"x":10.182864488674735,"y":0}]}],"marks":[{"type":"line","from":{"data":"fd662279-81b0-49bc-8f2a-57f78e64608b"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"y":{"scale":"y","field":"data.y"},"interpolate":{"value":"step-before"},"fill":{"value":"steelblue"},"fillOpacity":{"value":0.4},"stroke":{"value":"steelblue"},"strokeWidth":{"value":2},"strokeOpacity":{"value":1}}}}],"scales":[{"name":"x","type":"linear","range":"width","zero":false,"domain":{"data":"fd662279-81b0-49bc-8f2a-57f78e64608b","field":"data.x"}},{"name":"y","type":"linear","range":"height","nice":true,"zero":false,"domain":{"data":"fd662279-81b0-49bc-8f2a-57f78e64608b","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y","title":"uc-max","titleOffset":45}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"fd662279-81b0-49bc-8f2a-57f78e64608b\", :values ({:x 9.995690640909293, :y 0} {:x 10.006700867248437, :y 89.33512987858293} {:x 10.01771109358758, :y 0.0} {:x 10.028721319926724, :y 0.0} {:x 10.039731546265868, :y 0.0} {:x 10.050741772605011, :y 0.0} {:x 10.061751998944155, :y 0.0} {:x 10.072762225283299, :y 0.0} {:x 10.083772451622442, :y 0.0} {:x 10.094782677961586, :y 0.0} {:x 10.10579290430073, :y 0.0} {:x 10.116803130639873, :y 0.0} {:x 10.127813356979017, :y 0.0} {:x 10.13882358331816, :y 0.0} {:x 10.149833809657304, :y 0.0} {:x 10.160844035996448, :y 0.0} {:x 10.171854262335591, :y 1.4895243290044329} {:x 10.182864488674735, :y 0})}], :marks [{:type \"line\", :from {:data \"fd662279-81b0-49bc-8f2a-57f78e64608b\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :y {:scale \"y\", :field \"data.y\"}, :interpolate {:value \"step-before\"}, :fill {:value \"steelblue\"}, :fillOpacity {:value 0.4}, :stroke {:value \"steelblue\"}, :strokeWidth {:value 2}, :strokeOpacity {:value 1}}}}], :scales [{:name \"x\", :type \"linear\", :range \"width\", :zero false, :domain {:data \"fd662279-81b0-49bc-8f2a-57f78e64608b\", :field \"data.x\"}} {:name \"y\", :type \"linear\", :range \"height\", :nice true, :zero false, :domain {:data \"fd662279-81b0-49bc-8f2a-57f78e64608b\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\", :title \"uc-max\", :titleOffset 45}]}}"}
;; <=

;; @@

;; @@
