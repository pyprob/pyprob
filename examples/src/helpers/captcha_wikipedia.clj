(ns helpers.captcha-wikipedia
  (:require [helpers.captcha :refer [reduce-dim]]
            [anglican.runtime :refer :all])
  (:import [robots.OxCaptcha OxCaptcha]))

;; Renderer setup
(def WIDTH 200) ; Width of CAPTCHA
(def HEIGHT 50) ; Height of CAPTCHA
(def oxCaptcha (OxCaptcha. WIDTH HEIGHT)) ; Renderer object
(def fonts ["Courier 10 Pitch" "Nimbus Mono L"])

(defn render [letters font-size kerning]
  (.background oxCaptcha)
  (.setFont oxCaptcha (rand-nth fonts) 0 font-size)
  (.textCentered oxCaptcha (char-array letters) (int-array (repeat (count letters) 1)) (int kerning))
  (.distortionElastic oxCaptcha 14)
  (.distortionShear2 oxCaptcha (rand-int WIDTH) (+ 10 (rand-int 30)) (+ -7 (rand-int 14)) (rand-int HEIGHT) (+ 5 (rand-int 15)) (+ -6 (rand-int 12)))
  (.distortionElectric2 oxCaptcha)
  (.recenter oxCaptcha)
  (.blurGaussian oxCaptcha 0.5)
  (mapv #(into [] %) (seq (.getImageArray2D oxCaptcha))))

(defn render-to-file [letters font-size kerning filename]
  (.background oxCaptcha)
  (.setFont oxCaptcha (rand-nth fonts) 0 font-size)
  (.textCentered oxCaptcha (char-array letters) (int-array (repeat (count letters) 1)) (int kerning))
  (.distortionElastic oxCaptcha 14)
  (.distortionShear2 oxCaptcha (rand-int WIDTH) (+ 10 (rand-int 30)) (+ -7 (rand-int 14)) (rand-int HEIGHT) (+ 5 (rand-int 15)) (+ -6 (rand-int 12)))
  (.distortionElectric2 oxCaptcha)
  (.recenter oxCaptcha)
  (.blurGaussian oxCaptcha 0.5)
  (.save oxCaptcha filename))

;; Model specific
(def abc-sigma 1) ; Standard deviation calculated from each pixel (pixels range from 0 to 255)
(def letter-dict "abcdefghijklmnopqrstuvwxyz") ; Dictionary mapping from letter-id to letter
(defdist abc-dist
  "Approximate Bayesian Computation distribution constructor. Reduces dimensions using Random projections and then calculates likelihood under diagonal multivariate Gaussian under the rendered image."
  [rendered-image abc-sigma]
  [dist (normal 0 abc-sigma)]
  (sample* [this] rendered-image)
  (observe* [this baseline-image]
            (reduce + (map #(observe* dist (- %1 %2))
                           (reduce-dim baseline-image)
                           (reduce-dim rendered-image)))))
