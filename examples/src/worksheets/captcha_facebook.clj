;; gorilla-repl.fileformat = 1

;; **
;;; # Facebook's Captcha
;;; 
;;; In this worksheet, we demonstrate inference compilation on a Captcha rendering generative model and use it to break Facebook's Captcha. More in [the paper](https://arxiv.org/abs/1610.09900).
;; **

;; @@
(ns captcha-facebook
  (:require [gorilla-plot.core :as plot]
            anglican.rmh
            anglican.csis.csis
            anglican.smc
            [anglican.csis.network :refer :all]
            [anglican.inference :refer [infer]]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-facebook :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
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

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-facebook/combine-observes-fn</span>","value":"#'captcha-facebook/combine-observes-fn"}
;; <=

;; @@
;; Start the Torch connection
(def torch-connection (start-torch-connection captcha-facebook [nil] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-facebook/torch-connection</span>","value":"#'captcha-facebook/torch-connection"}
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
;;; ### Load real Facebook Captchas
;; **

;; @@
(def directory (clojure.java.io/file "resources/facebook-dataset"))
(def files (take 100 (filter #(= ".png" (apply str (take-last 4 (.getPath %))))
                             (rest (file-seq directory)))))
(def num-observes (count files))
(def observes (doall (map vec (map (fn [f] (map vec (.load oxCaptcha (.getPath f)))) files))))
(def ground-truth-letters (map (fn [f] (clojure.string/replace (.getName f) ".png" "")) files))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;captcha-facebook/ground-truth-letters</span>","value":"#'captcha-facebook/ground-truth-letters"}
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
                            (take num-particles (infer :smc captcha-facebook [observe] :number-of-particles num-particles)))
                          observes))
(def smc-MAP-list (map (comp empirical-MAP collect-results) smc-states-list))
(time 
  (doall (map (fn [smc-MAP filename] (render-to-file (:letters smc-MAP) (:font-size smc-MAP) (:kerning smc-MAP) filename))
              smc-MAP-list
              (map #(str "plots/captcha-facebook/" % "-smc.png") (range 1 (inc (count observes)))))))

;; RMH
(def num-iters 1)
(def rmh-states-list (map (fn [observe]
                            (take num-iters (infer :rmh captcha-facebook [observe])))
                          observes))
(def rmh-posterior-list (map (comp first last collect-results) rmh-states-list))
(time 
  (doall (map (fn [rmh-posterior filename] (render-to-file (:letters rmh-posterior) (:font-size rmh-posterior) (:kerning rmh-posterior) filename))
              rmh-posterior-list
              (map #(str "plots/captcha-facebook/" % "-rmh.png") (range 1 (inc (count observes)))))))

;; CSIS
(def num-particles 1)
(def csis-states-list (map (fn [observe]
                             (take num-particles (infer :csis captcha-facebook [observe])))
                           observes))
(def csis-MAP-list (map (comp empirical-MAP collect-results) csis-states-list))
(time 
  (doall (map (fn [csis-MAP filename] (render-to-file (:letters csis-MAP) (:font-size csis-MAP) (:kerning csis-MAP) filename))
              csis-MAP-list
              (map #(str "plots/captcha-facebook/" % "-csis.png") (range 1 (inc (count observes)))))))
;; @@
;; ->
;;; &quot;Elapsed time: 6989.973 msecs&quot;
;;; &quot;Elapsed time: 8368.194 msecs&quot;
;;; &quot;Elapsed time: 8157.594 msecs&quot;
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
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-string'>&quot;SwTIfLH&quot;</span>","value":"\"SwTIfLH\""},{"type":"html","content":"<span class='clj-string'>&quot;v9TcNOH&quot;</span>","value":"\"v9TcNOH\""},{"type":"html","content":"<span class='clj-string'>&quot;iyTtDOH&quot;</span>","value":"\"iyTtDOH\""},{"type":"html","content":"<span class='clj-string'>&quot;KpTIfvT&quot;</span>","value":"\"KpTIfvT\""},{"type":"html","content":"<span class='clj-string'>&quot;xZTMTsF&quot;</span>","value":"\"xZTMTsF\""},{"type":"html","content":"<span class='clj-string'>&quot;Sa2IfvH&quot;</span>","value":"\"Sa2IfvH\""},{"type":"html","content":"<span class='clj-string'>&quot;Cp2I4lT&quot;</span>","value":"\"Cp2I4lT\""},{"type":"html","content":"<span class='clj-string'>&quot;1QTMftR&quot;</span>","value":"\"1QTMftR\""},{"type":"html","content":"<span class='clj-string'>&quot;ywTtfLR&quot;</span>","value":"\"ywTtfLR\""},{"type":"html","content":"<span class='clj-string'>&quot;epbI4MH&quot;</span>","value":"\"epbI4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;xp2IfqH&quot;</span>","value":"\"xp2IfqH\""},{"type":"html","content":"<span class='clj-string'>&quot;SpTMfvH&quot;</span>","value":"\"SpTMfvH\""},{"type":"html","content":"<span class='clj-string'>&quot;ep2tfOF&quot;</span>","value":"\"ep2tfOF\""},{"type":"html","content":"<span class='clj-string'>&quot;Bp2M4MT&quot;</span>","value":"\"Bp2M4MT\""},{"type":"html","content":"<span class='clj-string'>&quot;epTt4LH&quot;</span>","value":"\"epTt4LH\""},{"type":"html","content":"<span class='clj-string'>&quot;CZ2cNwH&quot;</span>","value":"\"CZ2cNwH\""},{"type":"html","content":"<span class='clj-string'>&quot;VZ2MfLe&quot;</span>","value":"\"VZ2MfLe\""},{"type":"html","content":"<span class='clj-string'>&quot;Bw2MfOT&quot;</span>","value":"\"Bw2MfOT\""},{"type":"html","content":"<span class='clj-string'>&quot;BZTMfv7&quot;</span>","value":"\"BZTMfv7\""},{"type":"html","content":"<span class='clj-string'>&quot;y92MfsH&quot;</span>","value":"\"y92MfsH\""},{"type":"html","content":"<span class='clj-string'>&quot;ypBI4MH&quot;</span>","value":"\"ypBI4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;BZTc4sH&quot;</span>","value":"\"BZTc4sH\""},{"type":"html","content":"<span class='clj-string'>&quot;4w2M4M&quot;</span>","value":"\"4w2M4M\""},{"type":"html","content":"<span class='clj-string'>&quot;dw2IfvR&quot;</span>","value":"\"dw2IfvR\""},{"type":"html","content":"<span class='clj-string'>&quot;3pbp4MH&quot;</span>","value":"\"3pbp4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;VwTIflF&quot;</span>","value":"\"VwTIflF\""},{"type":"html","content":"<span class='clj-string'>&quot;SZTMfve&quot;</span>","value":"\"SZTMfve\""},{"type":"html","content":"<span class='clj-string'>&quot;xZTIfOR&quot;</span>","value":"\"xZTIfOR\""},{"type":"html","content":"<span class='clj-string'>&quot;Bm2IfvH&quot;</span>","value":"\"Bm2IfvH\""},{"type":"html","content":"<span class='clj-string'>&quot;V1TIfCR&quot;</span>","value":"\"V1TIfCR\""},{"type":"html","content":"<span class='clj-string'>&quot;1ZTI4lH&quot;</span>","value":"\"1ZTI4lH\""},{"type":"html","content":"<span class='clj-string'>&quot;SyTMfOT&quot;</span>","value":"\"SyTMfOT\""},{"type":"html","content":"<span class='clj-string'>&quot;SpTMfCR&quot;</span>","value":"\"SpTMfCR\""},{"type":"html","content":"<span class='clj-string'>&quot;ypTMNMH&quot;</span>","value":"\"ypTMNMH\""},{"type":"html","content":"<span class='clj-string'>&quot;yZTMfvR&quot;</span>","value":"\"yZTMfvR\""},{"type":"html","content":"<span class='clj-string'>&quot;LwTc4lH&quot;</span>","value":"\"LwTc4lH\""},{"type":"html","content":"<span class='clj-string'>&quot;VZTMfvF&quot;</span>","value":"\"VZTMfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;S32IfvH&quot;</span>","value":"\"S32IfvH\""},{"type":"html","content":"<span class='clj-string'>&quot;gp2IfcY&quot;</span>","value":"\"gp2IfcY\""},{"type":"html","content":"<span class='clj-string'>&quot;BZ2IfsF&quot;</span>","value":"\"BZ2IfsF\""},{"type":"html","content":"<span class='clj-string'>&quot;BaTRNLH&quot;</span>","value":"\"BaTRNLH\""},{"type":"html","content":"<span class='clj-string'>&quot;Sabs4CH&quot;</span>","value":"\"Sabs4CH\""},{"type":"html","content":"<span class='clj-string'>&quot;SpTMfvs&quot;</span>","value":"\"SpTMfvs\""},{"type":"html","content":"<span class='clj-string'>&quot;e32Rf9X&quot;</span>","value":"\"e32Rf9X\""},{"type":"html","content":"<span class='clj-string'>&quot;eyTM4MH&quot;</span>","value":"\"eyTM4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;Lp2M4MT&quot;</span>","value":"\"Lp2M4MT\""},{"type":"html","content":"<span class='clj-string'>&quot;4pTMfcC&quot;</span>","value":"\"4pTMfcC\""},{"type":"html","content":"<span class='clj-string'>&quot;4pTMfCe&quot;</span>","value":"\"4pTMfCe\""},{"type":"html","content":"<span class='clj-string'>&quot;ep2I4sC&quot;</span>","value":"\"ep2I4sC\""},{"type":"html","content":"<span class='clj-string'>&quot;Sw5IfvT&quot;</span>","value":"\"Sw5IfvT\""},{"type":"html","content":"<span class='clj-string'>&quot;BZTMSMf&quot;</span>","value":"\"BZTMSMf\""},{"type":"html","content":"<span class='clj-string'>&quot;Vp2cTXF&quot;</span>","value":"\"Vp2cTXF\""},{"type":"html","content":"<span class='clj-string'>&quot;xm2M4sT&quot;</span>","value":"\"xm2M4sT\""},{"type":"html","content":"<span class='clj-string'>&quot;VpTMfvF&quot;</span>","value":"\"VpTMfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;LQ2IfCH&quot;</span>","value":"\"LQ2IfCH\""},{"type":"html","content":"<span class='clj-string'>&quot;ByTIfsz&quot;</span>","value":"\"ByTIfsz\""},{"type":"html","content":"<span class='clj-string'>&quot;BpbtNwH&quot;</span>","value":"\"BpbtNwH\""},{"type":"html","content":"<span class='clj-string'>&quot;ByTc4Mf&quot;</span>","value":"\"ByTc4Mf\""},{"type":"html","content":"<span class='clj-string'>&quot;yp2ITOz&quot;</span>","value":"\"yp2ITOz\""},{"type":"html","content":"<span class='clj-string'>&quot;dyTIfvF&quot;</span>","value":"\"dyTIfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;xy2Mflf&quot;</span>","value":"\"xy2Mflf\""},{"type":"html","content":"<span class='clj-string'>&quot;eaTIf9f&quot;</span>","value":"\"eaTIf9f\""},{"type":"html","content":"<span class='clj-string'>&quot;xyTMf3H&quot;</span>","value":"\"xyTMf3H\""},{"type":"html","content":"<span class='clj-string'>&quot;VpTI4MT&quot;</span>","value":"\"VpTI4MT\""},{"type":"html","content":"<span class='clj-string'>&quot;yZTM4LT&quot;</span>","value":"\"yZTM4LT\""},{"type":"html","content":"<span class='clj-string'>&quot;BZTc4CH&quot;</span>","value":"\"BZTc4CH\""},{"type":"html","content":"<span class='clj-string'>&quot;VwTIfsH&quot;</span>","value":"\"VwTIfsH\""},{"type":"html","content":"<span class='clj-string'>&quot;QwTMfCF&quot;</span>","value":"\"QwTMfCF\""},{"type":"html","content":"<span class='clj-string'>&quot;BpTMEOT&quot;</span>","value":"\"BpTMEOT\""},{"type":"html","content":"<span class='clj-string'>&quot;VwTIfvF&quot;</span>","value":"\"VwTIfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;ypDIfvF&quot;</span>","value":"\"ypDIfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;SwxI4MH&quot;</span>","value":"\"SwxI4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;BpTMNlT&quot;</span>","value":"\"BpTMNlT\""},{"type":"html","content":"<span class='clj-string'>&quot;VpTMEOT&quot;</span>","value":"\"VpTMEOT\""},{"type":"html","content":"<span class='clj-string'>&quot;CwTMTsH&quot;</span>","value":"\"CwTMTsH\""},{"type":"html","content":"<span class='clj-string'>&quot;Bw2tfLH&quot;</span>","value":"\"Bw2tfLH\""},{"type":"html","content":"<span class='clj-string'>&quot;Cm2Mf9F&quot;</span>","value":"\"Cm2Mf9F\""},{"type":"html","content":"<span class='clj-string'>&quot;iZTIf9H&quot;</span>","value":"\"iZTIf9H\""},{"type":"html","content":"<span class='clj-string'>&quot;SZTIfOT&quot;</span>","value":"\"SZTIfOT\""},{"type":"html","content":"<span class='clj-string'>&quot;SyTIfvF&quot;</span>","value":"\"SyTIfvF\""},{"type":"html","content":"<span class='clj-string'>&quot;xwTM49H&quot;</span>","value":"\"xwTM49H\""},{"type":"html","content":"<span class='clj-string'>&quot;Ca2Mfvf&quot;</span>","value":"\"Ca2Mfvf\""},{"type":"html","content":"<span class='clj-string'>&quot;Vp2M4MH&quot;</span>","value":"\"Vp2M4MH\""},{"type":"html","content":"<span class='clj-string'>&quot;yZxIftH&quot;</span>","value":"\"yZxIftH\""},{"type":"html","content":"<span class='clj-string'>&quot;BETIfrF&quot;</span>","value":"\"BETIfrF\""},{"type":"html","content":"<span class='clj-string'>&quot;Ca2M4lT&quot;</span>","value":"\"Ca2M4lT\""},{"type":"html","content":"<span class='clj-string'>&quot;VpTMf9F&quot;</span>","value":"\"VpTMf9F\""},{"type":"html","content":"<span class='clj-string'>&quot;Ca2ITM&quot;</span>","value":"\"Ca2ITM\""},{"type":"html","content":"<span class='clj-string'>&quot;SpTI4LF&quot;</span>","value":"\"SpTI4LF\""},{"type":"html","content":"<span class='clj-string'>&quot;ypuM4sT&quot;</span>","value":"\"ypuM4sT\""},{"type":"html","content":"<span class='clj-string'>&quot;ypTcUwT&quot;</span>","value":"\"ypTcUwT\""},{"type":"html","content":"<span class='clj-string'>&quot;4LTM4LH&quot;</span>","value":"\"4LTM4LH\""},{"type":"html","content":"<span class='clj-string'>&quot;CwTMeve&quot;</span>","value":"\"CwTMeve\""},{"type":"html","content":"<span class='clj-string'>&quot;KybMElH&quot;</span>","value":"\"KybMElH\""},{"type":"html","content":"<span class='clj-string'>&quot;nQTMf9H&quot;</span>","value":"\"nQTMf9H\""},{"type":"html","content":"<span class='clj-string'>&quot;1wTM4L&quot;</span>","value":"\"1wTM4L\""},{"type":"html","content":"<span class='clj-string'>&quot;Vw2Mf9s&quot;</span>","value":"\"Vw2Mf9s\""},{"type":"html","content":"<span class='clj-string'>&quot;VpTYfv7&quot;</span>","value":"\"VpTYfv7\""},{"type":"html","content":"<span class='clj-string'>&quot;BpTM4sT&quot;</span>","value":"\"BpTM4sT\""},{"type":"html","content":"<span class='clj-string'>&quot;Sw2IflF&quot;</span>","value":"\"Sw2IflF\""}],"value":"(\"SwTIfLH\" \"v9TcNOH\" \"iyTtDOH\" \"KpTIfvT\" \"xZTMTsF\" \"Sa2IfvH\" \"Cp2I4lT\" \"1QTMftR\" \"ywTtfLR\" \"epbI4MH\" \"xp2IfqH\" \"SpTMfvH\" \"ep2tfOF\" \"Bp2M4MT\" \"epTt4LH\" \"CZ2cNwH\" \"VZ2MfLe\" \"Bw2MfOT\" \"BZTMfv7\" \"y92MfsH\" \"ypBI4MH\" \"BZTc4sH\" \"4w2M4M\" \"dw2IfvR\" \"3pbp4MH\" \"VwTIflF\" \"SZTMfve\" \"xZTIfOR\" \"Bm2IfvH\" \"V1TIfCR\" \"1ZTI4lH\" \"SyTMfOT\" \"SpTMfCR\" \"ypTMNMH\" \"yZTMfvR\" \"LwTc4lH\" \"VZTMfvF\" \"S32IfvH\" \"gp2IfcY\" \"BZ2IfsF\" \"BaTRNLH\" \"Sabs4CH\" \"SpTMfvs\" \"e32Rf9X\" \"eyTM4MH\" \"Lp2M4MT\" \"4pTMfcC\" \"4pTMfCe\" \"ep2I4sC\" \"Sw5IfvT\" \"BZTMSMf\" \"Vp2cTXF\" \"xm2M4sT\" \"VpTMfvF\" \"LQ2IfCH\" \"ByTIfsz\" \"BpbtNwH\" \"ByTc4Mf\" \"yp2ITOz\" \"dyTIfvF\" \"xy2Mflf\" \"eaTIf9f\" \"xyTMf3H\" \"VpTI4MT\" \"yZTM4LT\" \"BZTc4CH\" \"VwTIfsH\" \"QwTMfCF\" \"BpTMEOT\" \"VwTIfvF\" \"ypDIfvF\" \"SwxI4MH\" \"BpTMNlT\" \"VpTMEOT\" \"CwTMTsH\" \"Bw2tfLH\" \"Cm2Mf9F\" \"iZTIf9H\" \"SZTIfOT\" \"SyTIfvF\" \"xwTM49H\" \"Ca2Mfvf\" \"Vp2M4MH\" \"yZxIftH\" \"BETIfrF\" \"Ca2M4lT\" \"VpTMf9F\" \"Ca2ITM\" \"SpTI4LF\" \"ypuM4sT\" \"ypTcUwT\" \"4LTM4LH\" \"CwTMeve\" \"KybMElH\" \"nQTMf9H\" \"1wTM4L\" \"Vw2Mf9s\" \"VpTYfv7\" \"BpTM4sT\" \"Sw2IflF\")"}
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
;;; SMC :  0.9830357152223588 
;;; RMH :  0.9709523820877075 
;;; CSIS:  0.9842857152223587
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
