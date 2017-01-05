;; gorilla-repl.fileformat = 1

;; **
;;; # Wikipedia's Captcha
;;; 
;;; In this worksheet, we demonstrate inference compilation on a Captcha rendering generative model and use it to break Wikipedia's Captcha. More in [the paper](https://arxiv.org/abs/1610.09900).
;; **

;; @@
(ns captcha-wikipedia
  (:require [gorilla-plot.core :as plot]
            anglican.rmh
            anglican.csis.csis
            anglican.smc
            [anglican.csis.network :refer :all]
            [anglican.inference :refer [infer]]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-wikipedia :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
            [helpers.general :refer [empirical-MAP]]
            [anglican.stat :refer [collect-results]]
            [clojure.core.matrix :as m]
            [gorilla-repl.image :as image])
  (:use [anglican emit runtime]
        [gorilla-plot core])
  (:import [robots.OxCaptcha OxCaptcha]
           [javax.imageio ImageIO]
           [java.io File]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; ## Captcha generative model
;; **

;; @@
;; CAPTCHA Query
(with-primitive-procedures [render abc-dist repeatedly]
  (defquery captcha-wikipedia [baseline-image]
    (let [;; Number of letters in CAPTCHA
          num-letters (sample "numletters" (uniform-discrete 8 11))
          font-size (sample "fontsize" (uniform-discrete 26 32))
          kerning (sample "kerning" (uniform-discrete 1 3))
          letter-ids (repeatedly num-letters #(sample "letterid" (uniform-discrete 0 (count letter-dict))))
          letters (apply str (map (partial nth letter-dict) letter-ids))

          ;; Render image using renderer from ...
          rendered-image (render letters font-size kerning)]

      ;; ABC observe
      (observe (abc-dist rendered-image abc-sigma) baseline-image)
      
      ;; Retunrs
      {:letters letters
       :font-size font-size
       :kerning kerning})))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-wikipedia/captcha-wikipedia</span>","value":"#'captcha-wikipedia/captcha-wikipedia"}
;; <=

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-wikipedia/combine-observes-fn</span>","value":"#'captcha-wikipedia/combine-observes-fn"}
;; <=

;; @@
;; Start the Torch connection
(def torch-connection (start-torch-connection captcha-wikipedia [nil] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-wikipedia/torch-connection</span>","value":"#'captcha-wikipedia/torch-connection"}
;; <=

;; **
;;; `th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4`
;; **

;; @@
(stop-torch-connection torch-connection)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;Torch connection terminated.&quot;</span>","value":"\"Torch connection terminated.\""}
;; <=

;; **
;;; ## Inference comparison
;;; ### Load real Wikipedia Captchas
;; **

;; @@
(def directory (clojure.java.io/file "resources/wikipedia-dataset"))
(def files (take 100 (rest (file-seq directory))))
(def num-observes (count files))
(def observes (doall (map vec (map (fn [f] (map vec (.load oxCaptcha (.getPath f)))) files))))
(def ground-truth-letters (map (fn [f] (clojure.string/replace (.getName f) ".png" "")) files))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-wikipedia/ground-truth-letters</span>","value":"#'captcha-wikipedia/ground-truth-letters"}
;; <=

;; **
;;; ### Perform inference using SMC, RMH and CSIS
;; **

;; @@
;; SMC
(def num-particles 1)
(def smc-states-list (map (fn [observe]
                            (take num-particles (infer :smc captcha-wikipedia [observe] :number-of-particles num-particles)))
                          observes))
(def smc-MAP-list (map (comp empirical-MAP collect-results) smc-states-list))
(time 
  (doall (map (fn [smc-MAP filename] (render-to-file (:letters smc-MAP) (:font-size smc-MAP) (:kerning smc-MAP) filename))
              smc-MAP-list
              (map #(str "plots/captcha-wikipedia/" % "-smc.png") (range 1 (inc (count observes)))))))

;; RMH
(def num-iters 1)
(def rmh-states-list (map (fn [observe]
                            (take num-iters (infer :rmh captcha-wikipedia [observe])))
                          observes))
(def rmh-posterior-list (map (comp first last collect-results) rmh-states-list))
(time 
  (doall (map (fn [rmh-posterior filename] (render-to-file (:letters rmh-posterior) (:font-size rmh-posterior) (:kerning rmh-posterior) filename))
              rmh-posterior-list
              (map #(str "plots/captcha-wikipedia/" % "-rmh.png") (range 1 (inc (count observes)))))))

;; CSIS
(def num-particles 1)
(def csis-states-list (map (fn [observe]
                             (take num-particles (infer :csis captcha-wikipedia [observe])))
                           observes))
(def csis-MAP-list (map (comp empirical-MAP collect-results) csis-states-list))
(time 
  (doall (map (fn [csis-MAP filename] (render-to-file (:letters csis-MAP) (:font-size csis-MAP) (:kerning csis-MAP) filename))
              csis-MAP-list
              (map #(str "plots/captcha-wikipedia/" % "-csis.png") (range 1 (inc (count observes)))))))
;; @@
;; ->
;;; &quot;Elapsed time: 5360.777 msecs&quot;
;;; &quot;Elapsed time: 7677.055 msecs&quot;
;;; &quot;Elapsed time: 8329.806 msecs&quot;
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"},{"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}],"value":"(nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil)"}
;; <=

;; **
;;; ### Compare accuracies
;;; #### Letters
;; **

;; @@
(def smc-letters (map :letters smc-MAP-list))
(def rmh-letters (map :letters rmh-posterior-list))
(def csis-letters (map :letters csis-MAP-list))
ground-truth-letters
smc-letters
rmh-letters
csis-letters
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-string'>&quot;ewbsfssjjl&quot;</span>","value":"\"ewbsfssjjl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabijreetl&quot;</span>","value":"\"eabijreetl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtftsbil&quot;</span>","value":"\"eabtftsbil\""},{"type":"html","content":"<span class='clj-string'>&quot;xyxcfcfxqf&quot;</span>","value":"\"xyxcfcfxqf\""},{"type":"html","content":"<span class='clj-string'>&quot;ipbtfveuil&quot;</span>","value":"\"ipbtfveuil\""},{"type":"html","content":"<span class='clj-string'>&quot;yybsfcfjtl&quot;</span>","value":"\"yybsfcfjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;npbtfvzjil&quot;</span>","value":"\"npbtfvzjil\""},{"type":"html","content":"<span class='clj-string'>&quot;npuifczevl&quot;</span>","value":"\"npuifczevl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabkjcejql&quot;</span>","value":"\"eabkjcejql\""},{"type":"html","content":"<span class='clj-string'>&quot;npmtfsfxil&quot;</span>","value":"\"npmtfsfxil\""},{"type":"html","content":"<span class='clj-string'>&quot;xabtfvfevl&quot;</span>","value":"\"xabtfvfevl\""},{"type":"html","content":"<span class='clj-string'>&quot;eagcjtexol&quot;</span>","value":"\"eagcjtexol\""},{"type":"html","content":"<span class='clj-string'>&quot;yybkjlfxil&quot;</span>","value":"\"yybkjlfxil\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfsfeif&quot;</span>","value":"\"epbtfsfeif\""},{"type":"html","content":"<span class='clj-string'>&quot;dabsfctjty&quot;</span>","value":"\"dabsfctjty\""},{"type":"html","content":"<span class='clj-string'>&quot;epwtfcejij&quot;</span>","value":"\"epwtfcejij\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbpjcfjql&quot;</span>","value":"\"ypbpjcfjql\""},{"type":"html","content":"<span class='clj-string'>&quot;eywcjbexil&quot;</span>","value":"\"eywcjbexil\""},{"type":"html","content":"<span class='clj-string'>&quot;eamkjceuil&quot;</span>","value":"\"eamkjceuil\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbkfcsbll&quot;</span>","value":"\"dpbkfcsbll\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtjvseil&quot;</span>","value":"\"epbtjvseil\""},{"type":"html","content":"<span class='clj-string'>&quot;npbcjrfbil&quot;</span>","value":"\"npbcjrfbil\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfcsjql&quot;</span>","value":"\"eabtfcsjql\""},{"type":"html","content":"<span class='clj-string'>&quot;eabcjtkbyf&quot;</span>","value":"\"eabcjtkbyf\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbsfcfuyl&quot;</span>","value":"\"ypbsfcfuyl\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbpftsxll&quot;</span>","value":"\"dpbpftsxll\""},{"type":"html","content":"<span class='clj-string'>&quot;nwbsjrteql&quot;</span>","value":"\"nwbsjrteql\""},{"type":"html","content":"<span class='clj-string'>&quot;dybtfsfell&quot;</span>","value":"\"dybtfsfell\""},{"type":"html","content":"<span class='clj-string'>&quot;dybofcejtl&quot;</span>","value":"\"dybofcejtl\""},{"type":"html","content":"<span class='clj-string'>&quot;eaxtfvejlr&quot;</span>","value":"\"eaxtfvejlr\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtevcujl&quot;</span>","value":"\"ypbtevcujl\""},{"type":"html","content":"<span class='clj-string'>&quot;ipbtylzbql&quot;</span>","value":"\"ipbtylzbql\""},{"type":"html","content":"<span class='clj-string'>&quot;ypgcjlsbrl&quot;</span>","value":"\"ypgcjlsbrl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtfqeedy&quot;</span>","value":"\"ypbtfqeedy\""},{"type":"html","content":"<span class='clj-string'>&quot;nwbtjsfbql&quot;</span>","value":"\"nwbtjsfbql\""},{"type":"html","content":"<span class='clj-string'>&quot;epxcycsbyl&quot;</span>","value":"\"epxcycsbyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfspjif&quot;</span>","value":"\"eabtfspjif\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfvteil&quot;</span>","value":"\"eabtfvteil\""},{"type":"html","content":"<span class='clj-string'>&quot;dabtflfutl&quot;</span>","value":"\"dabtflfutl\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfvexyf&quot;</span>","value":"\"epbtfvexyf\""},{"type":"html","content":"<span class='clj-string'>&quot;epbkjvfuif&quot;</span>","value":"\"epbkjvfuif\""},{"type":"html","content":"<span class='clj-string'>&quot;dpxtfvfxyl&quot;</span>","value":"\"dpxtfvfxyl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbcjsfuil&quot;</span>","value":"\"ypbcjsfuil\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfrfeol&quot;</span>","value":"\"eabtfrfeol\""},{"type":"html","content":"<span class='clj-string'>&quot;epbpjlsbij&quot;</span>","value":"\"epbpjlsbij\""},{"type":"html","content":"<span class='clj-string'>&quot;xmmtfvfjyl&quot;</span>","value":"\"xmmtfvfjyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eybtfcsbql&quot;</span>","value":"\"eybtfcsbql\""},{"type":"html","content":"<span class='clj-string'>&quot;eamcjcsbjl&quot;</span>","value":"\"eamcjcsbjl\""},{"type":"html","content":"<span class='clj-string'>&quot;xabkjrsbtr&quot;</span>","value":"\"xabkjrsbtr\""},{"type":"html","content":"<span class='clj-string'>&quot;xabpjrejlf&quot;</span>","value":"\"xabpjrejlf\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfvejil&quot;</span>","value":"\"eabtfvejil\""},{"type":"html","content":"<span class='clj-string'>&quot;dmbcjrexyl&quot;</span>","value":"\"dmbcjrexyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eybcjrfuol&quot;</span>","value":"\"eybcjrfuol\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtjrfeol&quot;</span>","value":"\"epbtjrfeol\""},{"type":"html","content":"<span class='clj-string'>&quot;vampjrfxql&quot;</span>","value":"\"vampjrfxql\""},{"type":"html","content":"<span class='clj-string'>&quot;dpgsftejil&quot;</span>","value":"\"dpgsftejil\""},{"type":"html","content":"<span class='clj-string'>&quot;yabtfsseql&quot;</span>","value":"\"yabtfsseql\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtjvebij&quot;</span>","value":"\"epbtjvebij\""},{"type":"html","content":"<span class='clj-string'>&quot;yyxofveeif&quot;</span>","value":"\"yyxofveeif\""},{"type":"html","content":"<span class='clj-string'>&quot;dyftfvseif&quot;</span>","value":"\"dyftfvseif\""},{"type":"html","content":"<span class='clj-string'>&quot;yymtfsfjtl&quot;</span>","value":"\"yymtfsfjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;ymxcfvfuol&quot;</span>","value":"\"ymxcfvfuol\""},{"type":"html","content":"<span class='clj-string'>&quot;xabpjcfjtl&quot;</span>","value":"\"xabpjcfjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;ewbtfvfbyl&quot;</span>","value":"\"ewbtfvfbyl\""},{"type":"html","content":"<span class='clj-string'>&quot;epxiftebif&quot;</span>","value":"\"epxiftebif\""},{"type":"html","content":"<span class='clj-string'>&quot;yybtfcfjil&quot;</span>","value":"\"yybtfcfjil\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbkjcseyl&quot;</span>","value":"\"xpbkjcseyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eyutjveuil&quot;</span>","value":"\"eyutjveuil\""},{"type":"html","content":"<span class='clj-string'>&quot;yabkfsejll&quot;</span>","value":"\"yabkfsejll\""},{"type":"html","content":"<span class='clj-string'>&quot;npbsfsejjl&quot;</span>","value":"\"npbsfsejjl\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbtfvfjij&quot;</span>","value":"\"dpbtfvfjij\""},{"type":"html","content":"<span class='clj-string'>&quot;yadsjsejlj&quot;</span>","value":"\"yadsjsejlj\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbcjctjil&quot;</span>","value":"\"ypbcjctjil\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfvebll&quot;</span>","value":"\"epbtfvebll\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtjcfjil&quot;</span>","value":"\"epbtjcfjil\""},{"type":"html","content":"<span class='clj-string'>&quot;nwxceqejol&quot;</span>","value":"\"nwxceqejol\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjrkbql&quot;</span>","value":"\"epbcjrkbql\""},{"type":"html","content":"<span class='clj-string'>&quot;eybsjwextl&quot;</span>","value":"\"eybsjwextl\""},{"type":"html","content":"<span class='clj-string'>&quot;nwbkjreulf&quot;</span>","value":"\"nwbkjreulf\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxtfczjlf&quot;</span>","value":"\"eyxtfczjlf\""},{"type":"html","content":"<span class='clj-string'>&quot;ywbsfcsjyl&quot;</span>","value":"\"ywbsfcsjyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xwmcjcpevl&quot;</span>","value":"\"xwmcjcpevl\""},{"type":"html","content":"<span class='clj-string'>&quot;ywgtfcfjyl&quot;</span>","value":"\"ywgtfcfjyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eybtfcsevy&quot;</span>","value":"\"eybtfcsevy\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjrebol&quot;</span>","value":"\"epbcjrebol\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbkjvfxqx&quot;</span>","value":"\"xpbkjvfxqx\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfvejvl&quot;</span>","value":"\"epbtfvejvl\""},{"type":"html","content":"<span class='clj-string'>&quot;ipbkjvejqf&quot;</span>","value":"\"ipbkjvejqf\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbtjteeqy&quot;</span>","value":"\"xpbtjteeqy\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtfvfbil&quot;</span>","value":"\"ypbtfvfbil\""},{"type":"html","content":"<span class='clj-string'>&quot;dpgtjsfjif&quot;</span>","value":"\"dpgtjsfjif\""},{"type":"html","content":"<span class='clj-string'>&quot;epmtflfjqx&quot;</span>","value":"\"epmtflfjqx\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfvsbiy&quot;</span>","value":"\"epbtfvsbiy\""},{"type":"html","content":"<span class='clj-string'>&quot;dpxpfvfxll&quot;</span>","value":"\"dpxpfvfxll\""},{"type":"html","content":"<span class='clj-string'>&quot;dwbtjvzxll&quot;</span>","value":"\"dwbtjvzxll\""},{"type":"html","content":"<span class='clj-string'>&quot;embsfvejyl&quot;</span>","value":"\"embsfvejyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbcjosjll&quot;</span>","value":"\"xpbcjosjll\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtftfuqx&quot;</span>","value":"\"ypbtftfuqx\""},{"type":"html","content":"<span class='clj-string'>&quot;xpxtfcfxll&quot;</span>","value":"\"xpxtfcfxll\""},{"type":"html","content":"<span class='clj-string'>&quot;dpxtfcsjil&quot;</span>","value":"\"dpxtfcsjil\""}],"value":"(\"ewbsfssjjl\" \"eabijreetl\" \"eabtftsbil\" \"xyxcfcfxqf\" \"ipbtfveuil\" \"yybsfcfjtl\" \"npbtfvzjil\" \"npuifczevl\" \"eabkjcejql\" \"npmtfsfxil\" \"xabtfvfevl\" \"eagcjtexol\" \"yybkjlfxil\" \"epbtfsfeif\" \"dabsfctjty\" \"epwtfcejij\" \"ypbpjcfjql\" \"eywcjbexil\" \"eamkjceuil\" \"dpbkfcsbll\" \"epbtjvseil\" \"npbcjrfbil\" \"eabtfcsjql\" \"eabcjtkbyf\" \"ypbsfcfuyl\" \"dpbpftsxll\" \"nwbsjrteql\" \"dybtfsfell\" \"dybofcejtl\" \"eaxtfvejlr\" \"ypbtevcujl\" \"ipbtylzbql\" \"ypgcjlsbrl\" \"ypbtfqeedy\" \"nwbtjsfbql\" \"epxcycsbyl\" \"eabtfspjif\" \"eabtfvteil\" \"dabtflfutl\" \"epbtfvexyf\" \"epbkjvfuif\" \"dpxtfvfxyl\" \"ypbcjsfuil\" \"eabtfrfeol\" \"epbpjlsbij\" \"xmmtfvfjyl\" \"eybtfcsbql\" \"eamcjcsbjl\" \"xabkjrsbtr\" \"xabpjrejlf\" \"eabtfvejil\" \"dmbcjrexyl\" \"eybcjrfuol\" \"epbtjrfeol\" \"vampjrfxql\" \"dpgsftejil\" \"yabtfsseql\" \"epbtjvebij\" \"yyxofveeif\" \"dyftfvseif\" \"yymtfsfjtl\" \"ymxcfvfuol\" \"xabpjcfjtl\" \"ewbtfvfbyl\" \"epxiftebif\" \"yybtfcfjil\" \"xpbkjcseyl\" \"eyutjveuil\" \"yabkfsejll\" \"npbsfsejjl\" \"dpbtfvfjij\" \"yadsjsejlj\" \"ypbcjctjil\" \"epbtfvebll\" \"epbtjcfjil\" \"nwxceqejol\" \"epbcjrkbql\" \"eybsjwextl\" \"nwbkjreulf\" \"eyxtfczjlf\" \"ywbsfcsjyl\" \"xwmcjcpevl\" \"ywgtfcfjyl\" \"eybtfcsevy\" \"epbcjrebol\" \"xpbkjvfxqx\" \"epbtfvejvl\" \"ipbkjvejqf\" \"xpbtjteeqy\" \"ypbtfvfbil\" \"dpgtjsfjif\" \"epmtflfjqx\" \"epbtfvsbiy\" \"dpxpfvfxll\" \"dwbtjvzxll\" \"embsfvejyl\" \"xpbcjosjll\" \"ypbtftfuqx\" \"xpxtfcfxll\" \"dpxtfcsjil\")"}
;; <=

;; **
;;; #### Recognition rates
;; **

;; @@
(def smc-rate (* 100 (/ (count (filter identity (map = ground-truth-letters smc-letters))) (count ground-truth-letters))))
(def rmh-rate (* 100 (/ (count (filter identity (map = ground-truth-letters rmh-letters))) (count ground-truth-letters))))
(def csis-rate (* 100 (/ (count (filter identity (map = ground-truth-letters csis-letters))) (count ground-truth-letters))))
(println "SMC : " smc-rate "%\nRMH : " rmh-rate "%\nCSIS: " csis-rate "%")
;; @@
;; ->
;;; SMC :  0 %
;;; RMH :  0 %
;;; CSIS:  0 %
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; #### Levenshtein distances
;; **

;; @@
(def smc-levenshtein (mean (map levenshtein-normalized ground-truth-letters smc-letters)))
(def rmh-levenshtein (mean (map levenshtein-normalized ground-truth-letters rmh-letters)))
(def csis-levenshtein (mean (map levenshtein-normalized ground-truth-letters csis-letters)))
(println "SMC : " smc-levenshtein "\nRMH : " rmh-levenshtein "\nCSIS: " csis-levenshtein)
;; @@
;; ->
;;; SMC :  0.9323055493831635 
;;; RMH :  0.9403333294391633 
;;; CSIS:  0.934999989271164
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=
