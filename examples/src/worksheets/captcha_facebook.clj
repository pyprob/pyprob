;; gorilla-repl.fileformat = 1

;; **
;;; # Facebook's Captcha
;;; 
;;; In this worksheet, we demonstrate inference compilation on a Captcha rendering generative model and use it to break Facebook's Captcha. More in [the paper](https://arxiv.org/abs/1610.09900).
;; **

;; @@
(ns worksheets.captcha-facebook
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer :all]
            [anglican.stat :as stat]
            [anglican.infcomp.zmq :as zmq]
            [anglican.inference :refer [infer]]
            [anglican.infcomp.prior :as prior]
            [gorilla-plot.core :as plt]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-facebook :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
            [helpers.general :refer [empirical-MAP]]
            anglican.rmh
            anglican.infcomp.csis
            anglican.smc
            anglican.infcomp.core)
  (:import [robots.OxCaptcha OxCaptcha]
           [javax.imageio ImageIO]
           [java.io File]))

(anglican.infcomp.core/reset-infcomp-addressing-scheme!)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--26877$fn--26878]</span>","value":"#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--26877$fn--26878]"}
;; <=

;; **
;;; ## Captcha generative model
;; **

;; @@
;; CAPTCHA Query
(with-primitive-procedures [render abc-dist repeatedly]
  (defquery captcha-facebook [baseline-image]
    (let [;; Number of letters in CAPTCHA
          num-letters (sample (uniform-discrete 6 8))
          font-size (sample (uniform-discrete 38 44))
          kerning (sample (uniform-discrete -2 2))
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
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-facebook/captcha-facebook</span>","value":"#'worksheets.captcha-facebook/captcha-facebook"}
;; <=

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-facebook/combine-observes-fn</span>","value":"#'worksheets.captcha-facebook/combine-observes-fn"}
;; <=

;; @@
(def replier (zmq/start-replier captcha-facebook [nil] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-facebook/replier</span>","value":"#'worksheets.captcha-facebook/replier"}
;; <=

;; @@
(zmq/stop-replier replier)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;ZMQ connection terminated.&quot;</span>","value":"\"ZMQ connection terminated.\""}
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
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-facebook/ground-truth-letters</span>","value":"#'worksheets.captcha-facebook/ground-truth-letters"}
;; <=

;; **
;;; ### Load synthetic Facebook Captchas
;; **

;; @@
(def num-observes 100)
(def samples-from-prior (take num-observes (prior/sample-from-prior captcha-facebook nil)))
(def observes (map (comp combine-observes-fn :observes) samples-from-prior))
(def ground-truth-letters (map (fn [smp]
                                 (let [latents (:samples smp)
                                       letter-ids (map :value (filter #(= (:sample-address %) "letterid") latents))
                                       letters (apply str (map (partial nth letter-dict) letter-ids))]
                                   letters))
                               samples-from-prior))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-facebook/ground-truth-letters</span>","value":"#'worksheets.captcha-facebook/ground-truth-letters"}
;; <=

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
                            (take num-particles (infer :smc captcha-facebook [observe] :number-of-particles num-particles)))
                          observes))
(def smc-MAP-list (map (comp empirical-MAP stat/collect-results) smc-states-list))
(time
  (doall (map (fn [smc-MAP filename] (render-to-file (:letters smc-MAP) (:font-size smc-MAP) (:kerning smc-MAP) filename))
              smc-MAP-list
              (map #(str "plots/captcha-facebook/" % "-smc.png") (range 1 (inc (count observes)))))))

;; RMH
(def num-iters 1)
(def rmh-states-list (map (fn [observe]
                            (take num-iters (infer :rmh captcha-facebook [observe])))
                          observes))
(def rmh-posterior-list (map (comp first last stat/collect-results) rmh-states-list))
(time
  (doall (map (fn [rmh-posterior filename] (render-to-file (:letters rmh-posterior) (:font-size rmh-posterior) (:kerning rmh-posterior) filename))
              rmh-posterior-list
              (map #(str "plots/captcha-facebook/" % "-rmh.png") (range 1 (inc (count observes)))))))

;; CSIS
(def num-particles 1)
(def csis-states-list (map (fn [observe]
                             (take num-particles (infer :csis captcha-facebook [observe])))
                           observes))
(def csis-MAP-list (map (comp empirical-MAP stat/collect-results) csis-states-list))
(time
  (doall (map (fn [csis-MAP filename] (render-to-file (:letters csis-MAP) (:font-size csis-MAP) (:kerning csis-MAP) filename))
              csis-MAP-list
              (map #(str "plots/captcha-facebook/" % "-csis.png") (range 1 (inc (count observes)))))))
;; @@
;; ->
;;; &quot;Elapsed time: 6028.446 msecs&quot;
;;; &quot;Elapsed time: 7452.944 msecs&quot;
;;; &quot;Elapsed time: 9189.533 msecs&quot;
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
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-string'>&quot;llFWlW&quot;</span>","value":"\"llFWlW\""},{"type":"html","content":"<span class='clj-string'>&quot;G53WW5&quot;</span>","value":"\"G53WW5\""},{"type":"html","content":"<span class='clj-string'>&quot;YW33gGK&quot;</span>","value":"\"YW33gGK\""},{"type":"html","content":"<span class='clj-string'>&quot;GiiiFRi&quot;</span>","value":"\"GiiiFRi\""},{"type":"html","content":"<span class='clj-string'>&quot;FFWt7F&quot;</span>","value":"\"FFWt7F\""},{"type":"html","content":"<span class='clj-string'>&quot;FFlgig&quot;</span>","value":"\"FFlgig\""},{"type":"html","content":"<span class='clj-string'>&quot;wEkRWWW&quot;</span>","value":"\"wEkRWWW\""},{"type":"html","content":"<span class='clj-string'>&quot;44444R3&quot;</span>","value":"\"44444R3\""},{"type":"html","content":"<span class='clj-string'>&quot;RWgWR4&quot;</span>","value":"\"RWgWR4\""},{"type":"html","content":"<span class='clj-string'>&quot;FK8Kw5&quot;</span>","value":"\"FK8Kw5\""},{"type":"html","content":"<span class='clj-string'>&quot;kgglkg&quot;</span>","value":"\"kgglkg\""},{"type":"html","content":"<span class='clj-string'>&quot;Fl333l&quot;</span>","value":"\"Fl333l\""},{"type":"html","content":"<span class='clj-string'>&quot;llllll&quot;</span>","value":"\"llllll\""},{"type":"html","content":"<span class='clj-string'>&quot;3i33333&quot;</span>","value":"\"3i33333\""},{"type":"html","content":"<span class='clj-string'>&quot;rg8gg8&quot;</span>","value":"\"rg8gg8\""},{"type":"html","content":"<span class='clj-string'>&quot;WWl4W4&quot;</span>","value":"\"WWl4W4\""},{"type":"html","content":"<span class='clj-string'>&quot;FlkFln&quot;</span>","value":"\"FlkFln\""},{"type":"html","content":"<span class='clj-string'>&quot;YWGlYG3&quot;</span>","value":"\"YWGlYG3\""},{"type":"html","content":"<span class='clj-string'>&quot;GWWliWg&quot;</span>","value":"\"GWWliWg\""},{"type":"html","content":"<span class='clj-string'>&quot;KlKlKK&quot;</span>","value":"\"KlKlKK\""},{"type":"html","content":"<span class='clj-string'>&quot;gkkkkIk&quot;</span>","value":"\"gkkkkIk\""},{"type":"html","content":"<span class='clj-string'>&quot;KKi4i3n&quot;</span>","value":"\"KKi4i3n\""},{"type":"html","content":"<span class='clj-string'>&quot;gg6iggi&quot;</span>","value":"\"gg6iggi\""},{"type":"html","content":"<span class='clj-string'>&quot;GiGattG&quot;</span>","value":"\"GiGattG\""},{"type":"html","content":"<span class='clj-string'>&quot;lWglll&quot;</span>","value":"\"lWglll\""},{"type":"html","content":"<span class='clj-string'>&quot;GGG4kG3&quot;</span>","value":"\"GGG4kG3\""},{"type":"html","content":"<span class='clj-string'>&quot;5ll3l3&quot;</span>","value":"\"5ll3l3\""},{"type":"html","content":"<span class='clj-string'>&quot;Olllll&quot;</span>","value":"\"Olllll\""},{"type":"html","content":"<span class='clj-string'>&quot;FlFFll&quot;</span>","value":"\"FlFFll\""},{"type":"html","content":"<span class='clj-string'>&quot;FFgg4k&quot;</span>","value":"\"FFgg4k\""},{"type":"html","content":"<span class='clj-string'>&quot;G4gg7i&quot;</span>","value":"\"G4gg7i\""},{"type":"html","content":"<span class='clj-string'>&quot;Y4WrW4&quot;</span>","value":"\"Y4WrW4\""},{"type":"html","content":"<span class='clj-string'>&quot;kWYiKlG&quot;</span>","value":"\"kWYiKlG\""},{"type":"html","content":"<span class='clj-string'>&quot;8iwgwWa&quot;</span>","value":"\"8iwgwWa\""},{"type":"html","content":"<span class='clj-string'>&quot;344444&quot;</span>","value":"\"344444\""},{"type":"html","content":"<span class='clj-string'>&quot;FFF34Ml&quot;</span>","value":"\"FFF34Ml\""},{"type":"html","content":"<span class='clj-string'>&quot;Kg3Kl44&quot;</span>","value":"\"Kg3Kl44\""},{"type":"html","content":"<span class='clj-string'>&quot;KKKKYK&quot;</span>","value":"\"KKKKYK\""},{"type":"html","content":"<span class='clj-string'>&quot;WWlWWW&quot;</span>","value":"\"WWlWWW\""},{"type":"html","content":"<span class='clj-string'>&quot;3GWi3l&quot;</span>","value":"\"3GWi3l\""},{"type":"html","content":"<span class='clj-string'>&quot;gllllll&quot;</span>","value":"\"gllllll\""},{"type":"html","content":"<span class='clj-string'>&quot;WWWWwW&quot;</span>","value":"\"WWWWwW\""},{"type":"html","content":"<span class='clj-string'>&quot;mFFFFM&quot;</span>","value":"\"mFFFFM\""},{"type":"html","content":"<span class='clj-string'>&quot;gggg8u5&quot;</span>","value":"\"gggg8u5\""},{"type":"html","content":"<span class='clj-string'>&quot;lrKKll&quot;</span>","value":"\"lrKKll\""},{"type":"html","content":"<span class='clj-string'>&quot;llFFFl&quot;</span>","value":"\"llFFFl\""},{"type":"html","content":"<span class='clj-string'>&quot;G3F334&quot;</span>","value":"\"G3F334\""},{"type":"html","content":"<span class='clj-string'>&quot;t343Ul&quot;</span>","value":"\"t343Ul\""},{"type":"html","content":"<span class='clj-string'>&quot;w3Gllg&quot;</span>","value":"\"w3Gllg\""},{"type":"html","content":"<span class='clj-string'>&quot;G33g33g&quot;</span>","value":"\"G33g33g\""},{"type":"html","content":"<span class='clj-string'>&quot;GGWagRU&quot;</span>","value":"\"GGWagRU\""},{"type":"html","content":"<span class='clj-string'>&quot;tgFlW1&quot;</span>","value":"\"tgFlW1\""},{"type":"html","content":"<span class='clj-string'>&quot;5W55FG&quot;</span>","value":"\"5W55FG\""},{"type":"html","content":"<span class='clj-string'>&quot;4lWuWGG&quot;</span>","value":"\"4lWuWGG\""},{"type":"html","content":"<span class='clj-string'>&quot;4l53gg&quot;</span>","value":"\"4l53gg\""},{"type":"html","content":"<span class='clj-string'>&quot;y83YWA&quot;</span>","value":"\"y83YWA\""},{"type":"html","content":"<span class='clj-string'>&quot;WgYaGGY&quot;</span>","value":"\"WgYaGGY\""},{"type":"html","content":"<span class='clj-string'>&quot;kgkgk6&quot;</span>","value":"\"kgkgk6\""},{"type":"html","content":"<span class='clj-string'>&quot;k3ll3k3&quot;</span>","value":"\"k3ll3k3\""},{"type":"html","content":"<span class='clj-string'>&quot;WiWiii&quot;</span>","value":"\"WiWiii\""},{"type":"html","content":"<span class='clj-string'>&quot;Y88KG3K&quot;</span>","value":"\"Y88KG3K\""},{"type":"html","content":"<span class='clj-string'>&quot;F5lFF4F&quot;</span>","value":"\"F5lFF4F\""},{"type":"html","content":"<span class='clj-string'>&quot;F4g4F3F&quot;</span>","value":"\"F4g4F3F\""},{"type":"html","content":"<span class='clj-string'>&quot;5KKKFK&quot;</span>","value":"\"5KKKFK\""},{"type":"html","content":"<span class='clj-string'>&quot;o3k4444&quot;</span>","value":"\"o3k4444\""},{"type":"html","content":"<span class='clj-string'>&quot;YWiYYF&quot;</span>","value":"\"YWiYYF\""},{"type":"html","content":"<span class='clj-string'>&quot;Ftw38g&quot;</span>","value":"\"Ftw38g\""},{"type":"html","content":"<span class='clj-string'>&quot;4uWWWWW&quot;</span>","value":"\"4uWWWWW\""},{"type":"html","content":"<span class='clj-string'>&quot;W5h8W8W&quot;</span>","value":"\"W5h8W8W\""},{"type":"html","content":"<span class='clj-string'>&quot;kkklWl&quot;</span>","value":"\"kkklWl\""},{"type":"html","content":"<span class='clj-string'>&quot;3lll3l&quot;</span>","value":"\"3lll3l\""},{"type":"html","content":"<span class='clj-string'>&quot;3wKK33&quot;</span>","value":"\"3wKK33\""},{"type":"html","content":"<span class='clj-string'>&quot;uWm5l54&quot;</span>","value":"\"uWm5l54\""},{"type":"html","content":"<span class='clj-string'>&quot;88U3O88&quot;</span>","value":"\"88U3O88\""},{"type":"html","content":"<span class='clj-string'>&quot;klK3KGG&quot;</span>","value":"\"klK3KGG\""},{"type":"html","content":"<span class='clj-string'>&quot;33GW33&quot;</span>","value":"\"33GW33\""},{"type":"html","content":"<span class='clj-string'>&quot;r3K3UF3&quot;</span>","value":"\"r3K3UF3\""},{"type":"html","content":"<span class='clj-string'>&quot;ulFg3W&quot;</span>","value":"\"ulFg3W\""},{"type":"html","content":"<span class='clj-string'>&quot;3Fm88F&quot;</span>","value":"\"3Fm88F\""},{"type":"html","content":"<span class='clj-string'>&quot;W7SoFF&quot;</span>","value":"\"W7SoFF\""},{"type":"html","content":"<span class='clj-string'>&quot;rmG3YWw&quot;</span>","value":"\"rmG3YWw\""},{"type":"html","content":"<span class='clj-string'>&quot;Fi5iiiK&quot;</span>","value":"\"Fi5iiiK\""},{"type":"html","content":"<span class='clj-string'>&quot;llFFgFF&quot;</span>","value":"\"llFFgFF\""},{"type":"html","content":"<span class='clj-string'>&quot;uKKuiu&quot;</span>","value":"\"uKKuiu\""},{"type":"html","content":"<span class='clj-string'>&quot;llllllg&quot;</span>","value":"\"llllllg\""},{"type":"html","content":"<span class='clj-string'>&quot;54K444&quot;</span>","value":"\"54K444\""},{"type":"html","content":"<span class='clj-string'>&quot;4FFFFF&quot;</span>","value":"\"4FFFFF\""},{"type":"html","content":"<span class='clj-string'>&quot;lgtgiig&quot;</span>","value":"\"lgtgiig\""},{"type":"html","content":"<span class='clj-string'>&quot;3W33l3&quot;</span>","value":"\"3W33l3\""},{"type":"html","content":"<span class='clj-string'>&quot;3333333&quot;</span>","value":"\"3333333\""},{"type":"html","content":"<span class='clj-string'>&quot;iiiiiin&quot;</span>","value":"\"iiiiiin\""},{"type":"html","content":"<span class='clj-string'>&quot;K3gmkk&quot;</span>","value":"\"K3gmkk\""},{"type":"html","content":"<span class='clj-string'>&quot;5gwlgwg&quot;</span>","value":"\"5gwlgwg\""},{"type":"html","content":"<span class='clj-string'>&quot;88KkkWY&quot;</span>","value":"\"88KkkWY\""},{"type":"html","content":"<span class='clj-string'>&quot;ggZgggg&quot;</span>","value":"\"ggZgggg\""},{"type":"html","content":"<span class='clj-string'>&quot;GgGlG4W&quot;</span>","value":"\"GgGlG4W\""},{"type":"html","content":"<span class='clj-string'>&quot;448F44&quot;</span>","value":"\"448F44\""},{"type":"html","content":"<span class='clj-string'>&quot;Kgkggg&quot;</span>","value":"\"Kgkggg\""},{"type":"html","content":"<span class='clj-string'>&quot;WiiWWWw&quot;</span>","value":"\"WiiWWWw\""},{"type":"html","content":"<span class='clj-string'>&quot;gk4bk4&quot;</span>","value":"\"gk4bk4\""}],"value":"(\"llFWlW\" \"G53WW5\" \"YW33gGK\" \"GiiiFRi\" \"FFWt7F\" \"FFlgig\" \"wEkRWWW\" \"44444R3\" \"RWgWR4\" \"FK8Kw5\" \"kgglkg\" \"Fl333l\" \"llllll\" \"3i33333\" \"rg8gg8\" \"WWl4W4\" \"FlkFln\" \"YWGlYG3\" \"GWWliWg\" \"KlKlKK\" \"gkkkkIk\" \"KKi4i3n\" \"gg6iggi\" \"GiGattG\" \"lWglll\" \"GGG4kG3\" \"5ll3l3\" \"Olllll\" \"FlFFll\" \"FFgg4k\" \"G4gg7i\" \"Y4WrW4\" \"kWYiKlG\" \"8iwgwWa\" \"344444\" \"FFF34Ml\" \"Kg3Kl44\" \"KKKKYK\" \"WWlWWW\" \"3GWi3l\" \"gllllll\" \"WWWWwW\" \"mFFFFM\" \"gggg8u5\" \"lrKKll\" \"llFFFl\" \"G3F334\" \"t343Ul\" \"w3Gllg\" \"G33g33g\" \"GGWagRU\" \"tgFlW1\" \"5W55FG\" \"4lWuWGG\" \"4l53gg\" \"y83YWA\" \"WgYaGGY\" \"kgkgk6\" \"k3ll3k3\" \"WiWiii\" \"Y88KG3K\" \"F5lFF4F\" \"F4g4F3F\" \"5KKKFK\" \"o3k4444\" \"YWiYYF\" \"Ftw38g\" \"4uWWWWW\" \"W5h8W8W\" \"kkklWl\" \"3lll3l\" \"3wKK33\" \"uWm5l54\" \"88U3O88\" \"klK3KGG\" \"33GW33\" \"r3K3UF3\" \"ulFg3W\" \"3Fm88F\" \"W7SoFF\" \"rmG3YWw\" \"Fi5iiiK\" \"llFFgFF\" \"uKKuiu\" \"llllllg\" \"54K444\" \"4FFFFF\" \"lgtgiig\" \"3W33l3\" \"3333333\" \"iiiiiin\" \"K3gmkk\" \"5gwlgwg\" \"88KkkWY\" \"ggZgggg\" \"GgGlG4W\" \"448F44\" \"Kgkggg\" \"WiiWWWw\" \"gk4bk4\")"}
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
;;; SMC :  0.9809523814916611 
;;; RMH :  0.9780952388048172 
;;; CSIS:  0.9769047629833222
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
