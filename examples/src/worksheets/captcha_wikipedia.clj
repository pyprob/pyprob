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
            anglican.infcomp.csis
            anglican.smc
            [anglican.infcomp.network :refer :all]
            [anglican.infcomp.prior :refer [sample-from-prior]]
            [anglican.inference :refer [infer]]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-wikipedia :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
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

;; @@
;; Start the Torch connection
(def torch-connection (start-torch-connection captcha-wikipedia [nil] combine-observes-fn))
;; @@

;; @@
(stop-torch-connection torch-connection)
;; @@

;; **
;;; ## Inference comparison
;;; ### Load real Wikipedia Captchas
;; **

;; @@
(def directory (clojure.java.io/file "resources/wikipedia-dataset"))
(def files (take 100 (filter #(= ".png" (apply str (take-last 4 (.getPath %))))
                             (rest (file-seq directory)))))
(def num-observes (count files))
(def observes (doall (map vec (map (fn [f] (map vec (.load oxCaptcha (.getPath f)))) files))))
(def ground-truth-letters (map (fn [f] (clojure.string/replace (.getName f) ".png" "")) files))
;; @@

;; **
;;; ### Load synthetic Wikipedia Captchas
;; **

;; @@
(def num-observes 100)
(def samples-from-prior (take num-observes (sample-from-prior captcha-facebook nil)))
(def observes (map (comp combine-observes-fn :observes) samples-from-prior))
(def ground-truth-letters (map (fn [smp]
                                 (let [latents (:samples smp)
                                       letter-ids (map :value (filter #(= (:sample-address %) "letterid") latents))
                                       letters (apply str (map (partial nth letter-dict) letter-ids))]
                                   letters))
                               samples-from-prior))
;; @@

;; **
;;; ### Perform inference using SMC, RMH and CSIS
;;;
;;; Run inference server
;;; ```
;;; python -m infcomp.infer
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
