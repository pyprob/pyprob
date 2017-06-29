;; gorilla-repl.fileformat = 1

;; **
;;; # Bayes nets (non-Dirac observation)
;;; 
;;; http://www.robots.ox.ac.uk/~fwood/anglican/examples/viewer/?worksheet=bayes-net
;; **

;; @@
(ns worksheets.bayes-net-non-dirac
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
(defquery bayes-net-non-dirac [sprinkler wet-grass] 
  (let [is-cloudy (sample (flip 0.5))

        is-raining (if is-cloudy
                     (sample (flip 0.8))
                     (sample (flip 0.2)))

        sprinkler-dist (if is-cloudy
                         (flip 0.1)
                         (flip 0.5))

        wet-grass-dist (cond 
                         (and sprinkler is-raining) (flip 0.99)
                         (and (not sprinkler) (not is-raining)) (flip 0.0)
                         :else (flip 0.9))]

    (observe sprinkler-dist sprinkler)
    (observe wet-grass-dist wet-grass)

    is-raining))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net-non-dirac/bayes-net-non-dirac</span>","value":"#'worksheets.bayes-net-non-dirac/bayes-net-non-dirac"}
;; <=

;; **
;;; ## Inference Compilation
;; **

;; @@
(defn combine-observes-fn [observes]
  (vec (map #(if (true? (:value %)) 1 0) observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net-non-dirac/combine-observes-fn</span>","value":"#'worksheets.bayes-net-non-dirac/combine-observes-fn"}
;; <=

;; @@
(prior/sample-observes-from-prior bayes-net-non-dirac [true true])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O6</span>","value":"O6"}],"value":"[:observe-address O6]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[:value 0]"}],"value":"{:time-index 0, :observe-address O6, :observe-instance 0, :value 0}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>1</span>","value":"1"}],"value":"[:time-index 1]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O5</span>","value":"O5"}],"value":"[:observe-address O5]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[:value 1]"}],"value":"{:time-index 1, :observe-address O5, :observe-instance 0, :value 1}"}],"value":"[{:time-index 0, :observe-address O6, :observe-instance 0, :value 0} {:time-index 1, :observe-address O5, :observe-instance 0, :value 1}]"}
;; <=

;; @@
(combine-observes-fn (prior/sample-observes-from-prior bayes-net-non-dirac [true true]))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[0 0]"}
;; <=

;; @@
(def replier (zmq/start-replier bayes-net-non-dirac [true true] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net-non-dirac/replier</span>","value":"#'worksheets.bayes-net-non-dirac/replier"}
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
(def test-observes [true true])
(def test-observes-embedder-input (map #(if % 1 0) test-observes))
(def num-particles 100)

(def smc-states (take num-particles (infer :smc bayes-net-non-dirac test-observes :number-of-particles num-particles)))
(def smc-empirical-distribution (stat/empirical-distribution (stat/collect-results smc-states)))
(plt/bar-chart [true false]
               (mapv #(get smc-empirical-distribution %) [true false])
               :plot-range [:all [0 1]])
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"598d11b9-379a-4300-a260-8c563c38cf86","values":[{"x":true,"y":0.31000000000000005},{"x":false,"y":0.6900000000000001}]}],"marks":[{"type":"rect","from":{"data":"598d11b9-379a-4300-a260-8c563c38cf86"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"598d11b9-379a-4300-a260-8c563c38cf86","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":[0,1]}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"598d11b9-379a-4300-a260-8c563c38cf86\", :values ({:x true, :y 0.31000000000000005} {:x false, :y 0.6900000000000001})}], :marks [{:type \"rect\", :from {:data \"598d11b9-379a-4300-a260-8c563c38cf86\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"598d11b9-379a-4300-a260-8c563c38cf86\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain [0 1]}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@
(def csis-states (take num-particles (infer :csis bayes-net-non-dirac test-observes :observe-embedder-input test-observes-embedder-input)))
(def csis-empirical-distribution (stat/empirical-distribution (stat/collect-results csis-states)))
(plt/bar-chart [true false]
               (mapv #(get csis-empirical-distribution %) [true false])
               :plot-range [:all [0 1]])
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"a18a4537-54f0-480c-bcc9-a2078660f543","values":[{"x":true,"y":0.3102435350577775},{"x":false,"y":0.6897564649422224}]}],"marks":[{"type":"rect","from":{"data":"a18a4537-54f0-480c-bcc9-a2078660f543"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"a18a4537-54f0-480c-bcc9-a2078660f543","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":[0,1]}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"a18a4537-54f0-480c-bcc9-a2078660f543\", :values ({:x true, :y 0.3102435350577775} {:x false, :y 0.6897564649422224})}], :marks [{:type \"rect\", :from {:data \"a18a4537-54f0-480c-bcc9-a2078660f543\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"a18a4537-54f0-480c-bcc9-a2078660f543\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain [0 1]}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@

;; @@
