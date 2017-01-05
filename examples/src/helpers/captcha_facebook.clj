(ns helpers.captcha-facebook
  (:require [helpers.captcha :refer [reduce-dim]]
            [anglican.runtime :refer :all])
  (:import [robots.OxCaptcha OxCaptcha]))

;; Renderer setup
(def WIDTH 200) ; Width of CAPTCHA
(def HEIGHT 50) ; Height of CAPTCHA
(def oxCaptcha (OxCaptcha. WIDTH HEIGHT)) ; Renderer object
(def fonts ["Hind Semibold" "Noto Sans" "Ubuntu"])
(defn render [letters font-size kerning]
  (.background oxCaptcha)
  (.setFont oxCaptcha (rand-nth fonts) 0 font-size)
  (.text oxCaptcha (char-array letters) (+ 2 (rand-int 13)) (+ 29 (rand-int 4)) 1 (int kerning))
  (.distortionElastic oxCaptcha (+ 10 (rand-int 10)) 11 6)
  (.distortionShear2 oxCaptcha (rand-int WIDTH) (+ 10 (rand-int 30)) (+ -4 (rand-int 8)) (rand-int HEIGHT) (+ 5 (rand-int 15)) (+ -5 (rand-int 10)))
  (.noiseStrokes oxCaptcha (+ 6 (rand-int 5)) (+ 1.1 (rand 0.5)))
  (.noiseEllipses oxCaptcha (+ 4 (rand-int 3)) (+ 1.1 (rand 0.5)))
  ;(.blurGaussian oxCaptcha 0.1)
  (.noiseWhiteGaussian oxCaptcha 10)
  (mapv #(into [] %) (seq (.getImageArray2D oxCaptcha))))

(defn render-to-file [letters font-size kerning filename]
  (.background oxCaptcha)
  (.setFont oxCaptcha (rand-nth fonts) 0 font-size)
  (.text oxCaptcha (char-array letters) (+ 2 (rand-int 13)) (+ 29 (rand-int 4)) 1 (int kerning))
  (.distortionElastic oxCaptcha (+ 10 (rand-int 10)) 11 6)
  (.distortionShear2 oxCaptcha (rand-int WIDTH) (+ 10 (rand-int 30)) (+ -4 (rand-int 8)) (rand-int HEIGHT) (+ 5 (rand-int 15)) (+ -5 (rand-int 10)))
  (.noiseStrokes oxCaptcha (+ 6 (rand-int 5)) (+ 1.1 (rand 0.5)))
  (.noiseEllipses oxCaptcha (+ 4 (rand-int 3)) (+ 1.1 (rand 0.5)))
  ;(.blurGaussian oxCaptcha 0.1)
  (.noiseWhiteGaussian oxCaptcha 10)
  (.save oxCaptcha filename))

;; Model specific
(def abc-sigma 1) ; Standard deviation calculated from each pixel (pixels range from 0 to 255)
(def letter-dict "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789") ; Dictionary mapping from letter-id to letter
(defdist abc-dist
  "Approximate Bayesian Computation distribution constructor. Reduces dimensions using Random projections and then calculates likelihood under diagonal multivariate Gaussian under the rendered image."
  [rendered-image abc-sigma]
  [dist (normal 0 abc-sigma)]
  (sample* [this] rendered-image)
  (observe* [this baseline-image]
            (reduce + (map #(observe* dist (- %1 %2))
                           (reduce-dim baseline-image)
                           (reduce-dim rendered-image)))))
