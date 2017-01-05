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
;;; 
;;; Run inference server
;;; ```
;;; th infer.lua --latest
;;; ```
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
;;; &quot;Elapsed time: 4531.761 msecs&quot;
;;; &quot;Elapsed time: 7383.061 msecs&quot;
;;; &quot;Elapsed time: 38747.81 msecs&quot;
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
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-string'>&quot;ypbsjvfeyl&quot;</span>","value":"\"ypbsjvfeyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xwbsysebjf&quot;</span>","value":"\"xwbsysebjf\""},{"type":"html","content":"<span class='clj-string'>&quot;epxtftpbvl&quot;</span>","value":"\"epxtftpbvl\""},{"type":"html","content":"<span class='clj-string'>&quot;dpftfcfbol&quot;</span>","value":"\"dpftfcfbol\""},{"type":"html","content":"<span class='clj-string'>&quot;ywbtjlejif&quot;</span>","value":"\"ywbtjlejif\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbcjcsxil&quot;</span>","value":"\"xpbcjcsxil\""},{"type":"html","content":"<span class='clj-string'>&quot;yybtfrsjql&quot;</span>","value":"\"yybtfrsjql\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtjvkbil&quot;</span>","value":"\"eabtjvkbil\""},{"type":"html","content":"<span class='clj-string'>&quot;xkbpjqfjvl&quot;</span>","value":"\"xkbpjqfjvl\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxcjmfbiy&quot;</span>","value":"\"eyxcjmfbiy\""},{"type":"html","content":"<span class='clj-string'>&quot;nyfojcfeil&quot;</span>","value":"\"nyfojcfeil\""},{"type":"html","content":"<span class='clj-string'>&quot;epxtfvfjol&quot;</span>","value":"\"epxtfvfjol\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbiflfbll&quot;</span>","value":"\"ypbiflfbll\""},{"type":"html","content":"<span class='clj-string'>&quot;eybtjvfjyl&quot;</span>","value":"\"eybtjvfjyl\""},{"type":"html","content":"<span class='clj-string'>&quot;yabtfqfxil&quot;</span>","value":"\"yabtfqfxil\""},{"type":"html","content":"<span class='clj-string'>&quot;yabcjrsbll&quot;</span>","value":"\"yabcjrsbll\""},{"type":"html","content":"<span class='clj-string'>&quot;nymijqsuyl&quot;</span>","value":"\"nymijqsuyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbtjqfuql&quot;</span>","value":"\"xpbtjqfuql\""},{"type":"html","content":"<span class='clj-string'>&quot;epbsjvfeil&quot;</span>","value":"\"epbsjvfeil\""},{"type":"html","content":"<span class='clj-string'>&quot;xpmtfcsjyl&quot;</span>","value":"\"xpmtfcsjyl\""},{"type":"html","content":"<span class='clj-string'>&quot;iwxtfccjif&quot;</span>","value":"\"iwxtfccjif\""},{"type":"html","content":"<span class='clj-string'>&quot;yaxtfvejol&quot;</span>","value":"\"yaxtfvejol\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbcjsfbtl&quot;</span>","value":"\"ypbcjsfbtl\""},{"type":"html","content":"<span class='clj-string'>&quot;eybsfcexrl&quot;</span>","value":"\"eybsfcexrl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbsjvfeix&quot;</span>","value":"\"ypbsjvfeix\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbcjrfjlf&quot;</span>","value":"\"dpbcjrfjlf\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbkjtfell&quot;</span>","value":"\"xpbkjtfell\""},{"type":"html","content":"<span class='clj-string'>&quot;eybcjrfbol&quot;</span>","value":"\"eybcjrfbol\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjwfjil&quot;</span>","value":"\"epbcjwfjil\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxofsexll&quot;</span>","value":"\"eyxofsexll\""},{"type":"html","content":"<span class='clj-string'>&quot;vpbsjlfuif&quot;</span>","value":"\"vpbsjlfuif\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtyceuil&quot;</span>","value":"\"epbtyceuil\""},{"type":"html","content":"<span class='clj-string'>&quot;dwstjvfuql&quot;</span>","value":"\"dwstjvfuql\""},{"type":"html","content":"<span class='clj-string'>&quot;ewbtjqfbyl&quot;</span>","value":"\"ewbtjqfbyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabcjwsbil&quot;</span>","value":"\"eabcjwsbil\""},{"type":"html","content":"<span class='clj-string'>&quot;ypgcjvejtl&quot;</span>","value":"\"ypgcjvejtl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtfccjif&quot;</span>","value":"\"ypbtfccjif\""},{"type":"html","content":"<span class='clj-string'>&quot;dpgpfsebtl&quot;</span>","value":"\"dpgpfsebtl\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtjrfbql&quot;</span>","value":"\"epbtjrfbql\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfqsbiy&quot;</span>","value":"\"epbtfqsbiy\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbtjtfbif&quot;</span>","value":"\"dpbtjtfbif\""},{"type":"html","content":"<span class='clj-string'>&quot;yavcjtfjil&quot;</span>","value":"\"yavcjtfjil\""},{"type":"html","content":"<span class='clj-string'>&quot;eabsfvseyf&quot;</span>","value":"\"eabsfvseyf\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjcsbyl&quot;</span>","value":"\"epbcjcsbyl\""},{"type":"html","content":"<span class='clj-string'>&quot;eawtfvejil&quot;</span>","value":"\"eawtfvejil\""},{"type":"html","content":"<span class='clj-string'>&quot;xpmtjveblf&quot;</span>","value":"\"xpmtjveblf\""},{"type":"html","content":"<span class='clj-string'>&quot;nabtjrzeil&quot;</span>","value":"\"nabtjrzeil\""},{"type":"html","content":"<span class='clj-string'>&quot;epbsfcsbij&quot;</span>","value":"\"epbsfcsbij\""},{"type":"html","content":"<span class='clj-string'>&quot;epxcjvebiy&quot;</span>","value":"\"epxcjvebiy\""},{"type":"html","content":"<span class='clj-string'>&quot;ypwcjsfjjf&quot;</span>","value":"\"ypwcjsfjjf\""},{"type":"html","content":"<span class='clj-string'>&quot;eabkjcfjjj&quot;</span>","value":"\"eabkjcfjjj\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbsyvfeil&quot;</span>","value":"\"ypbsyvfeil\""},{"type":"html","content":"<span class='clj-string'>&quot;eamsfvsjil&quot;</span>","value":"\"eamsfvsjil\""},{"type":"html","content":"<span class='clj-string'>&quot;iabtfvsbif&quot;</span>","value":"\"iabtfvsbif\""},{"type":"html","content":"<span class='clj-string'>&quot;eabcjtebif&quot;</span>","value":"\"eabcjtebif\""},{"type":"html","content":"<span class='clj-string'>&quot;epmsfvebql&quot;</span>","value":"\"epmsfvebql\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtyqseif&quot;</span>","value":"\"epbtyqseif\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjrfbol&quot;</span>","value":"\"epbcjrfbol\""},{"type":"html","content":"<span class='clj-string'>&quot;eaxcfcfjwl&quot;</span>","value":"\"eaxcfcfjwl\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbsjcsbtx&quot;</span>","value":"\"xpbsjcsbtx\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbsfvsjil&quot;</span>","value":"\"dpbsfvsjil\""},{"type":"html","content":"<span class='clj-string'>&quot;epfcjvfeyl&quot;</span>","value":"\"epfcjvfeyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xydsjrseyl&quot;</span>","value":"\"xydsjrseyl\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbcywejil&quot;</span>","value":"\"xpbcywejil\""},{"type":"html","content":"<span class='clj-string'>&quot;xabtjskbil&quot;</span>","value":"\"xabtjskbil\""},{"type":"html","content":"<span class='clj-string'>&quot;yybtjsfjtl&quot;</span>","value":"\"yybtjsfjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;damtfvfuyl&quot;</span>","value":"\"damtfvfuyl\""},{"type":"html","content":"<span class='clj-string'>&quot;nybkevfbty&quot;</span>","value":"\"nybkevfbty\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtyqeeil&quot;</span>","value":"\"epbtyqeeil\""},{"type":"html","content":"<span class='clj-string'>&quot;eyftfrexlf&quot;</span>","value":"\"eyftfrexlf\""},{"type":"html","content":"<span class='clj-string'>&quot;dabtfvfuil&quot;</span>","value":"\"dabtfvfuil\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtflsbil&quot;</span>","value":"\"ypbtflsbil\""},{"type":"html","content":"<span class='clj-string'>&quot;xpxtfrexql&quot;</span>","value":"\"xpxtfrexql\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbcfvfbif&quot;</span>","value":"\"ypbcfvfbif\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxsfvsjtl&quot;</span>","value":"\"eyxsfvsjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbsfteeil&quot;</span>","value":"\"dpbsfteeil\""},{"type":"html","content":"<span class='clj-string'>&quot;dabtjvsjtl&quot;</span>","value":"\"dabtjvsjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;xaqtjsejil&quot;</span>","value":"\"xaqtjsejil\""},{"type":"html","content":"<span class='clj-string'>&quot;xpbsfsfbql&quot;</span>","value":"\"xpbsfsfbql\""},{"type":"html","content":"<span class='clj-string'>&quot;emssfqseif&quot;</span>","value":"\"emssfqseif\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxcfrtbll&quot;</span>","value":"\"eyxcfrtbll\""},{"type":"html","content":"<span class='clj-string'>&quot;eabsfsfeql&quot;</span>","value":"\"eabsfsfeql\""},{"type":"html","content":"<span class='clj-string'>&quot;xpfcjsfeij&quot;</span>","value":"\"xpfcjsfeij\""},{"type":"html","content":"<span class='clj-string'>&quot;epgkjlfjtl&quot;</span>","value":"\"epgkjlfjtl\""},{"type":"html","content":"<span class='clj-string'>&quot;eabtfvfbif&quot;</span>","value":"\"eabtfvfbif\""},{"type":"html","content":"<span class='clj-string'>&quot;epbtfsfjif&quot;</span>","value":"\"epbtfsfjif\""},{"type":"html","content":"<span class='clj-string'>&quot;eyxofskjqf&quot;</span>","value":"\"eyxofskjqf\""},{"type":"html","content":"<span class='clj-string'>&quot;eafcjrfjql&quot;</span>","value":"\"eafcjrfjql\""},{"type":"html","content":"<span class='clj-string'>&quot;epbcjrfbvl&quot;</span>","value":"\"epbcjrfbvl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbsfvsbyj&quot;</span>","value":"\"ypbsfvsbyj\""},{"type":"html","content":"<span class='clj-string'>&quot;ywutfcfjif&quot;</span>","value":"\"ywutfcfjif\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbcjwfbol&quot;</span>","value":"\"dpbcjwfbol\""},{"type":"html","content":"<span class='clj-string'>&quot;xyftfqeujl&quot;</span>","value":"\"xyftfqeujl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbtfvsetx&quot;</span>","value":"\"ypbtfvsetx\""},{"type":"html","content":"<span class='clj-string'>&quot;iputflfjll&quot;</span>","value":"\"iputflfjll\""},{"type":"html","content":"<span class='clj-string'>&quot;dabcjrfjvf&quot;</span>","value":"\"dabcjrfjvf\""},{"type":"html","content":"<span class='clj-string'>&quot;nybkjrebif&quot;</span>","value":"\"nybkjrebif\""},{"type":"html","content":"<span class='clj-string'>&quot;eaxtfsfeil&quot;</span>","value":"\"eaxtfsfeil\""},{"type":"html","content":"<span class='clj-string'>&quot;dpbtflsbjl&quot;</span>","value":"\"dpbtflsbjl\""},{"type":"html","content":"<span class='clj-string'>&quot;ypbsjtsbyj&quot;</span>","value":"\"ypbsjtsbyj\""}],"value":"(\"ypbsjvfeyl\" \"xwbsysebjf\" \"epxtftpbvl\" \"dpftfcfbol\" \"ywbtjlejif\" \"xpbcjcsxil\" \"yybtfrsjql\" \"eabtjvkbil\" \"xkbpjqfjvl\" \"eyxcjmfbiy\" \"nyfojcfeil\" \"epxtfvfjol\" \"ypbiflfbll\" \"eybtjvfjyl\" \"yabtfqfxil\" \"yabcjrsbll\" \"nymijqsuyl\" \"xpbtjqfuql\" \"epbsjvfeil\" \"xpmtfcsjyl\" \"iwxtfccjif\" \"yaxtfvejol\" \"ypbcjsfbtl\" \"eybsfcexrl\" \"ypbsjvfeix\" \"dpbcjrfjlf\" \"xpbkjtfell\" \"eybcjrfbol\" \"epbcjwfjil\" \"eyxofsexll\" \"vpbsjlfuif\" \"epbtyceuil\" \"dwstjvfuql\" \"ewbtjqfbyl\" \"eabcjwsbil\" \"ypgcjvejtl\" \"ypbtfccjif\" \"dpgpfsebtl\" \"epbtjrfbql\" \"epbtfqsbiy\" \"dpbtjtfbif\" \"yavcjtfjil\" \"eabsfvseyf\" \"epbcjcsbyl\" \"eawtfvejil\" \"xpmtjveblf\" \"nabtjrzeil\" \"epbsfcsbij\" \"epxcjvebiy\" \"ypwcjsfjjf\" \"eabkjcfjjj\" \"ypbsyvfeil\" \"eamsfvsjil\" \"iabtfvsbif\" \"eabcjtebif\" \"epmsfvebql\" \"epbtyqseif\" \"epbcjrfbol\" \"eaxcfcfjwl\" \"xpbsjcsbtx\" \"dpbsfvsjil\" \"epfcjvfeyl\" \"xydsjrseyl\" \"xpbcywejil\" \"xabtjskbil\" \"yybtjsfjtl\" \"damtfvfuyl\" \"nybkevfbty\" \"epbtyqeeil\" \"eyftfrexlf\" \"dabtfvfuil\" \"ypbtflsbil\" \"xpxtfrexql\" \"ypbcfvfbif\" \"eyxsfvsjtl\" \"dpbsfteeil\" \"dabtjvsjtl\" \"xaqtjsejil\" \"xpbsfsfbql\" \"emssfqseif\" \"eyxcfrtbll\" \"eabsfsfeql\" \"xpfcjsfeij\" \"epgkjlfjtl\" \"eabtfvfbif\" \"epbtfsfjif\" \"eyxofskjqf\" \"eafcjrfjql\" \"epbcjrfbvl\" \"ypbsfvsbyj\" \"ywutfcfjif\" \"dpbcjwfbol\" \"xyftfqeujl\" \"ypbtfvsetx\" \"iputflfjll\" \"dabcjrfjvf\" \"nybkjrebif\" \"eaxtfsfeil\" \"dpbtflsbjl\" \"ypbsjtsbyj\")"}
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
;;; SMC :  0.9393888849020005 
;;; RMH :  0.9351944398880006 
;;; CSIS:  0.9379999923706055
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
