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
(defquery uniform-continuous-query [obs]
  (let [uc-min (sample (normal 0 0.1))
        uc-range (sample (normal 10 0.1))
        uc-max (+ uc-min uc-range)
        mean (sample (uniform-continuous uc-min uc-max))
        std 0.01]
    (observe (normal mean std) obs)
    {:mean mean
     :uc-min uc-min
     :uc-max uc-max}))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous/uniform-continuous-query</span>","value":"#'worksheets.uniform-continuous/uniform-continuous-query"}
;; <=

;; **
;;; ## Inference Compilation
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous/combine-observes-fn</span>","value":"#'worksheets.uniform-continuous/combine-observes-fn"}
;; <=

;; @@
(prior/sample-observes-from-prior uniform-continuous-query [1])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O4</span>","value":"O4"}],"value":"[:observe-address O4]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-double'>3.7229471604222097</span>","value":"3.7229471604222097"}],"value":"[:value 3.7229471604222097]"}],"value":"{:time-index 0, :observe-address O4, :observe-instance 0, :value 3.7229471604222097}"}],"value":"[{:time-index 0, :observe-address O4, :observe-instance 0, :value 3.7229471604222097}]"}
;; <=

;; @@
(def replier (zmq/start-replier uniform-continuous-query [1] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.uniform-continuous/replier</span>","value":"#'worksheets.uniform-continuous/replier"}
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
(def csis-states (take num-particles (infer :csis uniform-continuous-query [5])))
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :mean :result) csis-states)))))) :normalize :probability-density :y-title "mean")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-min :result) csis-states)))))) :normalize :probability-density :y-title "uc-min")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-max :result) csis-states)))))) :normalize :probability-density :y-title "uc-max")
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"b706e8fa-7b61-42f0-bf20-697ce7d58ff5","values":[{"x":9.883359706782274,"y":0},{"x":9.89925321697403,"y":0.975240825531467},{"x":9.915146727165785,"y":0.0},{"x":9.931040237357541,"y":0.0},{"x":9.946933747549297,"y":0.0},{"x":9.962827257741052,"y":0.0},{"x":9.978720767932808,"y":0.0},{"x":9.994614278124564,"y":0.0},{"x":10.01050778831632,"y":46.9940240380292},{"x":10.026401298508075,"y":0.0},{"x":10.04229480869983,"y":0.0},{"x":10.058188318891586,"y":0.0},{"x":10.074081829083342,"y":0.0},{"x":10.089975339275098,"y":0.0},{"x":10.105868849466853,"y":0.0},{"x":10.121762359658609,"y":14.949498073953325},{"x":10.137655869850365,"y":0}]}],"marks":[{"type":"line","from":{"data":"b706e8fa-7b61-42f0-bf20-697ce7d58ff5"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"y":{"scale":"y","field":"data.y"},"interpolate":{"value":"step-before"},"fill":{"value":"steelblue"},"fillOpacity":{"value":0.4},"stroke":{"value":"steelblue"},"strokeWidth":{"value":2},"strokeOpacity":{"value":1}}}}],"scales":[{"name":"x","type":"linear","range":"width","zero":false,"domain":{"data":"b706e8fa-7b61-42f0-bf20-697ce7d58ff5","field":"data.x"}},{"name":"y","type":"linear","range":"height","nice":true,"zero":false,"domain":{"data":"b706e8fa-7b61-42f0-bf20-697ce7d58ff5","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y","title":"uc-max","titleOffset":45}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"b706e8fa-7b61-42f0-bf20-697ce7d58ff5\", :values ({:x 9.883359706782274, :y 0} {:x 9.89925321697403, :y 0.975240825531467} {:x 9.915146727165785, :y 0.0} {:x 9.931040237357541, :y 0.0} {:x 9.946933747549297, :y 0.0} {:x 9.962827257741052, :y 0.0} {:x 9.978720767932808, :y 0.0} {:x 9.994614278124564, :y 0.0} {:x 10.01050778831632, :y 46.9940240380292} {:x 10.026401298508075, :y 0.0} {:x 10.04229480869983, :y 0.0} {:x 10.058188318891586, :y 0.0} {:x 10.074081829083342, :y 0.0} {:x 10.089975339275098, :y 0.0} {:x 10.105868849466853, :y 0.0} {:x 10.121762359658609, :y 14.949498073953325} {:x 10.137655869850365, :y 0})}], :marks [{:type \"line\", :from {:data \"b706e8fa-7b61-42f0-bf20-697ce7d58ff5\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :y {:scale \"y\", :field \"data.y\"}, :interpolate {:value \"step-before\"}, :fill {:value \"steelblue\"}, :fillOpacity {:value 0.4}, :stroke {:value \"steelblue\"}, :strokeWidth {:value 2}, :strokeOpacity {:value 1}}}}], :scales [{:name \"x\", :type \"linear\", :range \"width\", :zero false, :domain {:data \"b706e8fa-7b61-42f0-bf20-697ce7d58ff5\", :field \"data.x\"}} {:name \"y\", :type \"linear\", :range \"height\", :nice true, :zero false, :domain {:data \"b706e8fa-7b61-42f0-bf20-697ce7d58ff5\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\", :title \"uc-max\", :titleOffset 45}]}}"}
;; <=

;; @@
(def num-particles 100)
(def is-states (take num-particles (infer :importance uniform-continuous-query [5])))
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :mean :result) is-states)))))) :normalize :probability-density :y-title "mean")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-min :result) is-states)))))) :normalize :probability-density :y-title "uc-min")
(plt/histogram (repeatedly 10000 #(sample* (categorical (vec (stat/empirical-distribution (stat/collect-by (comp :uc-max :result) is-states)))))) :normalize :probability-density :y-title "uc-max")
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"0ec69cfd-0280-4a2a-a333-a6add3b18c1f","values":[{"x":9.816906134076014,"y":0},{"x":10.816906134076014,"y":1.0},{"x":11.816906134076014,"y":0}]}],"marks":[{"type":"line","from":{"data":"0ec69cfd-0280-4a2a-a333-a6add3b18c1f"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"y":{"scale":"y","field":"data.y"},"interpolate":{"value":"step-before"},"fill":{"value":"steelblue"},"fillOpacity":{"value":0.4},"stroke":{"value":"steelblue"},"strokeWidth":{"value":2},"strokeOpacity":{"value":1}}}}],"scales":[{"name":"x","type":"linear","range":"width","zero":false,"domain":{"data":"0ec69cfd-0280-4a2a-a333-a6add3b18c1f","field":"data.x"}},{"name":"y","type":"linear","range":"height","nice":true,"zero":false,"domain":{"data":"0ec69cfd-0280-4a2a-a333-a6add3b18c1f","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y","title":"uc-max","titleOffset":45}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"0ec69cfd-0280-4a2a-a333-a6add3b18c1f\", :values ({:x 9.816906134076014, :y 0} {:x 10.816906134076014, :y 1.0} {:x 11.816906134076014, :y 0})}], :marks [{:type \"line\", :from {:data \"0ec69cfd-0280-4a2a-a333-a6add3b18c1f\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :y {:scale \"y\", :field \"data.y\"}, :interpolate {:value \"step-before\"}, :fill {:value \"steelblue\"}, :fillOpacity {:value 0.4}, :stroke {:value \"steelblue\"}, :strokeWidth {:value 2}, :strokeOpacity {:value 1}}}}], :scales [{:name \"x\", :type \"linear\", :range \"width\", :zero false, :domain {:data \"0ec69cfd-0280-4a2a-a333-a6add3b18c1f\", :field \"data.x\"}} {:name \"y\", :type \"linear\", :range \"height\", :nice true, :zero false, :domain {:data \"0ec69cfd-0280-4a2a-a333-a6add3b18c1f\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\", :title \"uc-max\", :titleOffset 45}]}}"}
;; <=

;; @@

;; @@
