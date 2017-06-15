;; gorilla-repl.fileformat = 1

;; **
;;; # Probabilistic Deterministic Finite Automata
;; **

;; @@
(ns pdfa
  (:require anglican.infcomp.csis
            anglican.smc
            anglican.infcomp.core
            [anglican.infcomp.network :refer :all]
            [anglican.inference :refer [infer]]
            [anglican.stat :refer [empirical-distribution collect-predicts collect-by collect-results]]
            [helpers.gmm :refer [normalize]])
  (:use [anglican emit runtime]
        [gorilla-plot core]))

(anglican.infcomp.core/reset-infcomp-addressing-scheme!)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--27417$fn--27418]</span>","value":"#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--27417$fn--27418]"}
;; <=

;; **
;;; ## Model
;; **

;; @@
(with-primitive-procedures [normalize]
  (defquery pdfa [observedsequence alphabet]
    (let [K (sample (uniform-discrete 1 5))
          nextstatedist (discrete (repeat K 1))
          emissionprior (fn [] (normalize (repeatedly (count alphabet) (fn [] (sample (uniform-continuous 0 1))))))
          transition (mem (fn [state symbol] (sample nextstatedist)))
          emissiondist (mem (fn [state] (categorical (zipmap alphabet (emissionprior)))))
          N (count observedsequence)]
      (loop [obs observedsequence
             state [0]]
        (if (not (empty? obs))
          (let [currentstate (peek state)
                _ (observe (emissiondist currentstate) (first obs))
                nextstate (transition currentstate (first obs))]
            (recur (rest obs)
                   (conj state nextstate)))
          {:statesequence state})))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;pdfa/pdfa</span>","value":"#'pdfa/pdfa"}
;; <=

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(def alphabet [0 1 2 3 4])
(sample-observes-from-prior pdfa [(repeat 10 nil) alphabet])
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:time-index 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>0</span>","value":"0"}],"value":"[:observe-instance 0]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>4</span>","value":"4"}],"value":"[:value 4]"}],"value":"{:time-index 0, :observe-address O11, :observe-instance 0, :value 4}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>1</span>","value":"1"}],"value":"[:time-index 1]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>1</span>","value":"1"}],"value":"[:observe-instance 1]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>2</span>","value":"2"}],"value":"[:value 2]"}],"value":"{:time-index 1, :observe-address O11, :observe-instance 1, :value 2}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>2</span>","value":"2"}],"value":"[:time-index 2]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>2</span>","value":"2"}],"value":"[:observe-instance 2]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[:value 0]"}],"value":"{:time-index 2, :observe-address O11, :observe-instance 2, :value 0}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>3</span>","value":"3"}],"value":"[:time-index 3]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>3</span>","value":"3"}],"value":"[:observe-instance 3]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[:value 1]"}],"value":"{:time-index 3, :observe-address O11, :observe-instance 3, :value 1}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>4</span>","value":"4"}],"value":"[:time-index 4]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>4</span>","value":"4"}],"value":"[:observe-instance 4]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>3</span>","value":"3"}],"value":"[:value 3]"}],"value":"{:time-index 4, :observe-address O11, :observe-instance 4, :value 3}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>5</span>","value":"5"}],"value":"[:time-index 5]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>5</span>","value":"5"}],"value":"[:observe-instance 5]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>3</span>","value":"3"}],"value":"[:value 3]"}],"value":"{:time-index 5, :observe-address O11, :observe-instance 5, :value 3}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>6</span>","value":"6"}],"value":"[:time-index 6]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>6</span>","value":"6"}],"value":"[:observe-instance 6]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>3</span>","value":"3"}],"value":"[:value 3]"}],"value":"{:time-index 6, :observe-address O11, :observe-instance 6, :value 3}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>7</span>","value":"7"}],"value":"[:time-index 7]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>7</span>","value":"7"}],"value":"[:observe-instance 7]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[:value 0]"}],"value":"{:time-index 7, :observe-address O11, :observe-instance 7, :value 0}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>8</span>","value":"8"}],"value":"[:time-index 8]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>8</span>","value":"8"}],"value":"[:observe-instance 8]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}],"value":"[:value 1]"}],"value":"{:time-index 8, :observe-address O11, :observe-instance 8, :value 1}"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:time-index</span>","value":":time-index"},{"type":"html","content":"<span class='clj-unkown'>9</span>","value":"9"}],"value":"[:time-index 9]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-address</span>","value":":observe-address"},{"type":"html","content":"<span class='clj-symbol'>O11</span>","value":"O11"}],"value":"[:observe-address O11]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:observe-instance</span>","value":":observe-instance"},{"type":"html","content":"<span class='clj-unkown'>9</span>","value":"9"}],"value":"[:observe-instance 9]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:value</span>","value":":value"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[:value 0]"}],"value":"{:time-index 9, :observe-address O11, :observe-instance 9, :value 0}"}],"value":"[{:time-index 0, :observe-address O11, :observe-instance 0, :value 4} {:time-index 1, :observe-address O11, :observe-instance 1, :value 2} {:time-index 2, :observe-address O11, :observe-instance 2, :value 0} {:time-index 3, :observe-address O11, :observe-instance 3, :value 1} {:time-index 4, :observe-address O11, :observe-instance 4, :value 3} {:time-index 5, :observe-address O11, :observe-instance 5, :value 3} {:time-index 6, :observe-address O11, :observe-instance 6, :value 3} {:time-index 7, :observe-address O11, :observe-instance 7, :value 0} {:time-index 8, :observe-address O11, :observe-instance 8, :value 1} {:time-index 9, :observe-address O11, :observe-instance 9, :value 0}]"}
;; <=

;; @@
(defn combine-observes-fn [observes]
  (map :value observes))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;pdfa/combine-observes-fn</span>","value":"#'pdfa/combine-observes-fn"}
;; <=

;; @@
(def torch-connection (start-torch-connection pdfa [(repeat 10 nil) alphabet] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;pdfa/torch-connection</span>","value":"#'pdfa/torch-connection"}
;; <=

;; @@
;; Stop the Torch connection
(stop-torch-connection torch-connection)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;Torch connection terminated.&quot;</span>","value":"\"Torch connection terminated.\""}
;; <=

;; **
;;; ## Inference
;; **

;; @@
(def test-observations
  (combine-observes-fn (sample-observes-from-prior pdfa [(repeat 10 nil) alphabet])))
test-observations
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"(0 0 1 1 0 1 0 0 0 0)"}
;; <=

;; **
;;; ### Infcomp
;;; 
;;; First, run the inference server from torch-infcomp:
;;; 
;;; `python -m infcomp.infer`
;;; 
;;; Then run the query, specifying number of particles:
;; **

;; @@
(def num-particles 1000)
(def csis-states (take num-particles (infer :csis pdfa [test-observations alphabet])))
(repeatedly 1 #(sample* (categorical (vec (empirical-distribution (collect-results csis-states))))))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:statesequence</span>","value":":statesequence"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[0 0 0 0 0 0 0 0 0 0 0]"}],"value":"[:statesequence [0 0 0 0 0 0 0 0 0 0 0]]"}],"value":"{:statesequence [0 0 0 0 0 0 0 0 0 0 0]}"}],"value":"({:statesequence [0 0 0 0 0 0 0 0 0 0 0]})"}
;; <=

;; @@
(def num-particles 1000)
(def is-states (take num-particles (infer :importance pdfa [test-observations alphabet])))
(repeatedly 1 #(sample* (categorical (vec (empirical-distribution (collect-results is-states))))))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:statesequence</span>","value":":statesequence"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[0 0 0 0 0 0 0 0 0 0 0]"}],"value":"[:statesequence [0 0 0 0 0 0 0 0 0 0 0]]"}],"value":"{:statesequence [0 0 0 0 0 0 0 0 0 0 0]}"}],"value":"({:statesequence [0 0 0 0 0 0 0 0 0 0 0]})"}
;; <=

;; @@
(def num-particles 1000)
(def smc-states (take num-particles (infer :smc pdfa [test-observations alphabet] :number-of-particles 1000)))
(repeatedly 1 #(sample* (categorical (vec (empirical-distribution (collect-results smc-states))))))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:statesequence</span>","value":":statesequence"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"},{"type":"html","content":"<span class='clj-long'>0</span>","value":"0"}],"value":"[0 0 0 0 0 0 0 0 0 0 0]"}],"value":"[:statesequence [0 0 0 0 0 0 0 0 0 0 0]]"}],"value":"{:statesequence [0 0 0 0 0 0 0 0 0 0 0]}"}],"value":"({:statesequence [0 0 0 0 0 0 0 0 0 0 0]})"}
;; <=

;; @@

;; @@
