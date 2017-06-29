;; gorilla-repl.fileformat = 1

;; **
;;; # Bayes nets
;;; 
;;; http://www.robots.ox.ac.uk/~fwood/anglican/examples/viewer/?worksheet=bayes-net
;; **

;; @@
(ns worksheets.bayes-net
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
(defdist dirac
  "Dirac distribution centered on x"
  [x] []
  (sample* [this] x)
  (observe* [this value] 
            (if (= x value) 
              0.0  
              (- (/ 1.0 0.0)))))

(with-primitive-procedures [dirac]
  (defquery bayes-net [sprinkler-obs wet-grass-obs] 
    (let [is-cloudy (sample (flip 0.5))

          is-raining (if is-cloudy 
                       (sample (flip 0.8))
                       (sample (flip 0.2)))
          sprinkler  (if is-cloudy 
                       (sample (flip 0.1))
                       (sample (flip 0.5)))

          wet-grass  (cond (and sprinkler is-raining) (sample (flip 0.99))
                           (and (not sprinkler) (not is-raining)) (sample (flip 0.0))
                           :else (sample (flip 0.9)))]

      (observe (dirac sprinkler) sprinkler-obs)
      (observe (dirac wet-grass) wet-grass-obs)

      is-raining)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net/bayes-net</span>","value":"#'worksheets.bayes-net/bayes-net"}
;; <=

;; **
;;; ## Inference Compilation
;; **

;; @@
(defn combine-observes-fn [observes]
  (vec (map #(if (true? (:value %)) 1 0) observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net/combine-observes-fn</span>","value":"#'worksheets.bayes-net/combine-observes-fn"}
;; <=

;; @@
(prior/sample-observes-from-prior bayes-net [true true])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O13</span>","value":"O13"}],"value":"[:observe-address O13]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"}],"value":"[:value false]"}],"value":"{:time-index 0, :observe-address O13, :observe-instance 0, :value false}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>1</span>","value":"1"}],"value":"[:time-index 1]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O10</span>","value":"O10"}],"value":"[:observe-address O10]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"}],"value":"[:value true]"}],"value":"{:time-index 1, :observe-address O10, :observe-instance 0, :value true}"}],"value":"[{:time-index 0, :observe-address O13, :observe-instance 0, :value false} {:time-index 1, :observe-address O10, :observe-instance 0, :value true}]"}
;; <=

;; @@
(combine-observes-fn (prior/sample-observes-from-prior bayes-net [true true]))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[1 1]"}
;; <=

;; @@
(def replier (zmq/start-replier bayes-net nil combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.bayes-net/replier</span>","value":"#'worksheets.bayes-net/replier"}
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

(def smc-states (take num-particles (infer :smc bayes-net test-observes :number-of-particles num-particles)))
(def smc-empirical-distribution (stat/empirical-distribution (stat/collect-results smc-states)))
(plt/bar-chart [true false]
               (mapv #(get smc-empirical-distribution %) [true false])
               :plot-range [:all [0 1]])
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"4ffaacb5-4859-474e-b163-0fca0f8974b7","values":[{"x":true,"y":0.30000000000000004},{"x":false,"y":0.7}]}],"marks":[{"type":"rect","from":{"data":"4ffaacb5-4859-474e-b163-0fca0f8974b7"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"4ffaacb5-4859-474e-b163-0fca0f8974b7","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":[0,1]}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"4ffaacb5-4859-474e-b163-0fca0f8974b7\", :values ({:x true, :y 0.30000000000000004} {:x false, :y 0.7})}], :marks [{:type \"rect\", :from {:data \"4ffaacb5-4859-474e-b163-0fca0f8974b7\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"4ffaacb5-4859-474e-b163-0fca0f8974b7\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain [0 1]}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@
(def csis-states (take num-particles (infer :csis bayes-net test-observes :observe-embedder-input test-observes-embedder-input)))
(def csis-empirical-distribution (stat/empirical-distribution (stat/collect-results csis-states)))
(plt/bar-chart [true false]
               (mapv #(get csis-empirical-distribution %) [true false])
               :plot-range [:all [0 1]])
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"5b7dbb5a-8165-46c8-ab33-7c2285360c8b","values":[{"x":true,"y":0.3492818844243008},{"x":false,"y":0.6507181155756994}]}],"marks":[{"type":"rect","from":{"data":"5b7dbb5a-8165-46c8-ab33-7c2285360c8b"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"5b7dbb5a-8165-46c8-ab33-7c2285360c8b","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":[0,1]}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"5b7dbb5a-8165-46c8-ab33-7c2285360c8b\", :values ({:x true, :y 0.3492818844243008} {:x false, :y 0.6507181155756994})}], :marks [{:type \"rect\", :from {:data \"5b7dbb5a-8165-46c8-ab33-7c2285360c8b\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"5b7dbb5a-8165-46c8-ab33-7c2285360c8b\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain [0 1]}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@

;; @@
