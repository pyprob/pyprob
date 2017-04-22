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
            anglican.infcomp.csis
            anglican.smc
            [anglican.infcomp.network :refer :all]
            [anglican.inference :refer [infer]]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-facebook :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
            [helpers.general :refer [empirical-MAP]]
            [anglican.stat :refer [collect-results]])
  (:use [anglican emit runtime])
  (:import [robots.OxCaptcha OxCaptcha]
           [javax.imageio ImageIO]
           [java.io File]))
;; @@

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

;; **
;;; `th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4`
;; **

;; @@
(stop-torch-connection torch-connection)
;; @@

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

;; **
;;; #### Recognition rates
;; **

;; @@
(def smc-rate (* 100 (/ (count (filter identity (map = ground-truth-letters smc-letters))) (count ground-truth-letters))))
(def rmh-rate (* 100 (/ (count (filter identity (map = ground-truth-letters rmh-letters))) (count ground-truth-letters))))
(def csis-rate (* 100 (/ (count (filter identity (map = ground-truth-letters csis-letters))) (count ground-truth-letters))))
(println "SMC : " smc-rate "%\nRMH : " rmh-rate "%\nCSIS: " csis-rate "%")
;; @@

;; **
;;; #### Levenshtein distances
;; **

;; @@
(def smc-levenshtein (mean (map levenshtein-normalized ground-truth-letters smc-letters)))
(def rmh-levenshtein (mean (map levenshtein-normalized ground-truth-letters rmh-letters)))
(def csis-levenshtein (mean (map levenshtein-normalized ground-truth-letters csis-letters)))
(println "SMC : " smc-levenshtein "\nRMH : " rmh-levenshtein "\nCSIS: " csis-levenshtein)
;; @@
