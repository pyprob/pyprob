;; gorilla-repl.fileformat = 1

;; **
;;; # Bayes nets
;;; 
;;; http://www.robots.ox.ac.uk/~fwood/anglican/examples/viewer/?worksheet=bayes-net
;; **

;; @@
(ns bayes-net
  (:require [gorilla-plot.core :as plot]
            [anglican.stat :as s]
            [anglican.stat :refer [empirical-distribution collect-results]]
            anglican.infcomp.csis
            anglican.smc
            [anglican.infcomp.network :refer :all]
            [anglican.infcomp.prior :refer [sample-from-prior]]
            [anglican.inference :refer [infer]]
            [helpers.general :refer [empirical-MAP]]
            [anglican.stat :refer [collect-results]])
  (:use [anglican core emit runtime])
  (:import [javax.imageio ImageIO]
           [java.io File]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; ## Bayes net generative model
;; **

;; @@
;; CAPTCHA Query
(with-primitive-procedures [render abc-dist repeatedly]
  (defquery captcha-facebook [baseline-image]
    (let [;; Number of letters in CAPTCHA
          num-letters (sample "numletters" (uniform-discrete 6 8))
          font-size (sample "fontsize" (uniform-discrete 38 44))
          kerning (sample "kerning" (uniform-discrete -2 2))
          letter-ids (repeatedly num-letters #(sample "letterid" (uniform-discrete 0 (count letter-dict))))
          letters (apply str (map (partial nth letter-dict) letter-ids))

          ;; Render image using renderer from ...
          rendered-image (render letters font-size kerning)]

      ;; ABC-style observe
      (observe (abc-dist rendered-image abc-sigma) baseline-image)

      ;; Returns
      {:letters letters
       :font-size font-size
       :kerning kerning})))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-facebook/captcha-facebook</span>","value":"#'captcha-facebook/captcha-facebook"}
;; <=

;; @@
(defquery bayes-net [sprinkler wet-grass] 
  (let [is-cloudy (sample "is-cloudy" (flip 0.5))

        is-raining (cond 
                     (= is-cloudy true ) 
                     (sample "is-raining-a" (flip 0.8))
                     (= is-cloudy false) 
                     (sample "is-raining-b" (flip 0.2)))

        sprinkler-dist  (cond 
                          (= is-cloudy true ) 
                          (flip 0.1)
                          (= is-cloudy false) 
                          (flip 0.5))

        wet-grass-dist  (cond 
                          (and (= sprinkler true) 
                               (= is-raining true))
                          (flip 0.99)
                          (and (= sprinkler false) 
                               (= is-raining false))
                          (flip 0.0)
                          (or  (= sprinkler true) 
                               (= is-raining true))
                          (flip 0.9))]

    (observe sprinkler-dist sprinkler)
    (observe wet-grass-dist wet-grass)


    is-raining))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;bayes-net/bayes-net</span>","value":"#'bayes-net/bayes-net"}
;; <=

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(defn combine-observes-fn [observes]
  (vec (map #(if (true? (:value %)) 1 0) observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;bayes-net/combine-observes-fn</span>","value":"#'bayes-net/combine-observes-fn"}
;; <=

;; @@
(sample-observes-from-prior bayes-net [true true])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O27767</span>","value":"O27767"}],"value":"[:observe-address O27767]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"}],"value":"[:value true]"}],"value":"{:time-index 0, :observe-address O27767, :observe-instance 0, :value true}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>1</span>","value":"1"}],"value":"[:time-index 1]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O27766</span>","value":"O27766"}],"value":"[:observe-address O27766]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"}],"value":"[:value true]"}],"value":"{:time-index 1, :observe-address O27766, :observe-instance 0, :value true}"}],"value":"[{:time-index 0, :observe-address O27767, :observe-instance 0, :value true} {:time-index 1, :observe-address O27766, :observe-instance 0, :value true}]"}
;; <=

;; @@
(combine-observes-fn (sample-observes-from-prior bayes-net [true true]))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[1 1]"}
;; <=

;; @@
;; Start the Torch connection
(def torch-connection (start-torch-connection bayes-net [true true] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;bayes-net/torch-connection</span>","value":"#'bayes-net/torch-connection"}
;; <=

;; @@
(stop-torch-connection torch-connection)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;Torch connection terminated.&quot;</span>","value":"\"Torch connection terminated.\""}
;; <=

;; @@
(->> (doquery :smc bayes-net [true true] :number-of-particles 10)
     (take 1000)
     s/collect-results
     (s/empirical-distribution)
     (#(plot/bar-chart (keys %) (vals %))))
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"910bc24a-faae-4382-a147-0e9410d07b41","values":[{"x":false,"y":0.6808727090905593},{"x":true,"y":0.3191272909094406}]}],"marks":[{"type":"rect","from":{"data":"910bc24a-faae-4382-a147-0e9410d07b41"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"910bc24a-faae-4382-a147-0e9410d07b41","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":{"data":"910bc24a-faae-4382-a147-0e9410d07b41","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"910bc24a-faae-4382-a147-0e9410d07b41\", :values ({:x false, :y 0.6808727090905593} {:x true, :y 0.3191272909094406})}], :marks [{:type \"rect\", :from {:data \"910bc24a-faae-4382-a147-0e9410d07b41\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"910bc24a-faae-4382-a147-0e9410d07b41\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain {:data \"910bc24a-faae-4382-a147-0e9410d07b41\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@
(def num-particles 100)
(def test-observes [true true])
(def csis-states (take num-particles (infer :csis bayes-net test-observes :observe-embedder-input (combine-observes-fn test-observes))))
(def posterior-empirical-dist (vec (s/empirical-distribution (collect-results csis-states))))
(plot/bar-chart (keys posterior-empirical-dist) (vals posterior-empirical-dist))
;; @@
;; =>
;;; {"type":"vega","content":{"width":400,"height":247.2187957763672,"padding":{"top":10,"left":55,"bottom":40,"right":10},"data":[{"name":"3009b5d8-8837-496f-bee6-894909cce9f5","values":[{"x":true,"y":0.28923316833589474},{"x":false,"y":0.7107668316641054}]}],"marks":[{"type":"rect","from":{"data":"3009b5d8-8837-496f-bee6-894909cce9f5"},"properties":{"enter":{"x":{"scale":"x","field":"data.x"},"width":{"scale":"x","band":true,"offset":-1},"y":{"scale":"y","field":"data.y"},"y2":{"scale":"y","value":0}},"update":{"fill":{"value":"steelblue"},"opacity":{"value":1}},"hover":{"fill":{"value":"#FF29D2"}}}}],"scales":[{"name":"x","type":"ordinal","range":"width","domain":{"data":"3009b5d8-8837-496f-bee6-894909cce9f5","field":"data.x"}},{"name":"y","range":"height","nice":true,"domain":{"data":"3009b5d8-8837-496f-bee6-894909cce9f5","field":"data.y"}}],"axes":[{"type":"x","scale":"x"},{"type":"y","scale":"y"}]},"value":"#gorilla_repl.vega.VegaView{:content {:width 400, :height 247.2188, :padding {:top 10, :left 55, :bottom 40, :right 10}, :data [{:name \"3009b5d8-8837-496f-bee6-894909cce9f5\", :values ({:x true, :y 0.28923316833589474} {:x false, :y 0.7107668316641054})}], :marks [{:type \"rect\", :from {:data \"3009b5d8-8837-496f-bee6-894909cce9f5\"}, :properties {:enter {:x {:scale \"x\", :field \"data.x\"}, :width {:scale \"x\", :band true, :offset -1}, :y {:scale \"y\", :field \"data.y\"}, :y2 {:scale \"y\", :value 0}}, :update {:fill {:value \"steelblue\"}, :opacity {:value 1}}, :hover {:fill {:value \"#FF29D2\"}}}}], :scales [{:name \"x\", :type \"ordinal\", :range \"width\", :domain {:data \"3009b5d8-8837-496f-bee6-894909cce9f5\", :field \"data.x\"}} {:name \"y\", :range \"height\", :nice true, :domain {:data \"3009b5d8-8837-496f-bee6-894909cce9f5\", :field \"data.y\"}}], :axes [{:type \"x\", :scale \"x\"} {:type \"y\", :scale \"y\"}]}}"}
;; <=

;; @@
posterior-empirical-dist
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"},{"type":"html","content":"<span class='clj-double'>0.28923316833589474</span>","value":"0.28923316833589474"}],"value":"[true 0.28923316833589474]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-double'>0.7107668316641054</span>","value":"0.7107668316641054"}],"value":"[false 0.7107668316641054]"}],"value":"[[true 0.28923316833589474] [false 0.7107668316641054]]"}
;; <=

;; @@
(frequencies (map :result csis-states))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"},{"type":"html","content":"<span class='clj-long'>45</span>","value":"45"}],"value":"[true 45]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-long'>55</span>","value":"55"}],"value":"[false 55]"}],"value":"{true 45, false 55}"}
;; <=

;; @@
(def num-particles 10)
(def test-observes [true true])
(def is-states (take num-particles (infer :importance bayes-net test-observes)))
(def posterior-empirical-dist (vec (s/empirical-distribution (collect-results is-states))))
(plot/bar-chart (keys posterior-empirical-dist) (vals posterior-empirical-dist))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;bayes-net/test-observes</span>","value":"#'bayes-net/test-observes"}
;; <=

;; @@
csis-states
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:anglican.state/mem</span>","value":":anglican.state/mem"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[],"value":"{}"}],"value":"[:anglican.state/mem {}]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:predicts</span>","value":":predicts"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[],"value":"[]"}],"value":"[:predicts []]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:anglican.infcomp.csis/samples</span>","value":":anglican.infcomp.csis/samples"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-address</span>","value":":sample-address"},{"type":"html","content":"<span class='clj-string'>&quot;is-cloudy&quot;</span>","value":"\"is-cloudy\""}],"value":"[:sample-address \"is-cloudy\"]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-instance</span>","value":":sample-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:sample-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-prior-dist</span>","value":":sample-prior-dist"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:p</span>","value":":p"},{"type":"html","content":"<span class='clj-double'>0.5</span>","value":"0.5"}],"value":"[:p 0.5]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:dist</span>","value":":dist"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:min</span>","value":":min"},{"type":"html","content":"<span class='clj-double'>0.0</span>","value":"0.0"}],"value":"[:min 0.0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:max</span>","value":":max"},{"type":"html","content":"<span class='clj-double'>1.0</span>","value":"1.0"}],"value":"[:max 1.0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:dist23748</span>","value":":dist23748"},{"type":"html","content":"<span class='clj-unkown'>#object[org.apache.commons.math3.distribution.UniformRealDistribution 0x6854b832 &quot;org.apache.commons.math3.distribution.UniformRealDistribution@6854b832&quot;]</span>","value":"#object[org.apache.commons.math3.distribution.UniformRealDistribution 0x6854b832 \"org.apache.commons.math3.distribution.UniformRealDistribution@6854b832\"]"}],"value":"[:dist23748 #object[org.apache.commons.math3.distribution.UniformRealDistribution 0x6854b832 \"org.apache.commons.math3.distribution.UniformRealDistribution@6854b832\"]]"}],"value":"(anglican.runtime/uniform-continuous 0.0 1.0)"}],"value":"[:dist (anglican.runtime/uniform-continuous 0.0 1.0)]"}],"value":"(anglican.runtime/flip 0.5)"}],"value":"[:sample-prior-dist (anglican.runtime/flip 0.5)]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[:value 1]"}],"value":"{:sample-address \"is-cloudy\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.5), :value 1}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-address</span>","value":":sample-address"},{"type":"html","content":"<span class='clj-string'>&quot;is-raining-a&quot;</span>","value":"\"is-raining-a\""}],"value":"[:sample-address \"is-raining-a\"]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-instance</span>","value":":sample-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:sample-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:sample-prior-dist</span>","value":":sample-prior-dist"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:p</span>","value":":p"},{"type":"html","content":"<span class='clj-double'>0.8</span>","value":"0.8"}],"value":"[:p 0.8]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:dist</span>","value":":dist"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:min</span>","value":":min"},{"type":"html","content":"<span class='clj-double'>0.0</span>","value":"0.0"}],"value":"[:min 0.0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:max</span>","value":":max"},{"type":"html","content":"<span class='clj-double'>1.0</span>","value":"1.0"}],"value":"[:max 1.0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:dist23748</span>","value":":dist23748"},{"type":"html","content":"<span class='clj-unkown'>#object[org.apache.commons.math3.distribution.UniformRealDistribution 0x596f040a &quot;org.apache.commons.math3.distribution.UniformRealDistribution@596f040a&quot;]</span>","value":"#object[org.apache.commons.math3.distribution.UniformRealDistribution 0x596f040a \"org.apache.commons.math3.distribution.UniformRealDistribution@596f040a\"]"}],"value":"[:dist23748 #object[org.apache.commons.math3.distribution.UniformRealDistribution 0x596f040a \"org.apache.commons.math3.distribution.UniformRealDistribution@596f040a\"]]"}],"value":"(anglican.runtime/uniform-continuous 0.0 1.0)"}],"value":"[:dist (anglican.runtime/uniform-continuous 0.0 1.0)]"}],"value":"(anglican.runtime/flip 0.8)"}],"value":"[:sample-prior-dist (anglican.runtime/flip 0.8)]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[:value 1]"}],"value":"{:sample-address \"is-raining-a\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.8), :value 1}"}],"value":"[{:sample-address \"is-cloudy\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.5), :value 1} {:sample-address \"is-raining-a\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.8), :value 1}]"}],"value":"[:anglican.infcomp.csis/samples [{:sample-address \"is-cloudy\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.5), :value 1} {:sample-address \"is-raining-a\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.8), :value 1}]]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:result</span>","value":":result"},{"type":"html","content":"<span class='clj-unkown'>true</span>","value":"true"}],"value":"[:result true]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:log-weight</span>","value":":log-weight"},{"type":"html","content":"<span class='clj-double'>-Infinity</span>","value":"-Infinity"}],"value":"[:log-weight -Infinity]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:anglican.state/store</span>","value":":anglican.state/store"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}],"value":"[:anglican.state/store nil]"}],"value":"{:anglican.state/mem {}, :predicts [], :anglican.infcomp.csis/samples [{:sample-address \"is-cloudy\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.5), :value 1} {:sample-address \"is-raining-a\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.8), :value 1}], :result true, :log-weight -Infinity, :anglican.state/store nil}"}],"value":"({:anglican.state/mem {}, :predicts [], :anglican.infcomp.csis/samples [{:sample-address \"is-cloudy\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.5), :value 1} {:sample-address \"is-raining-a\", :sample-instance 0, :sample-prior-dist (anglican.runtime/flip 0.8), :value 1}], :result true, :log-weight -Infinity, :anglican.state/store nil})"}
;; <=

;; @@

;; @@
