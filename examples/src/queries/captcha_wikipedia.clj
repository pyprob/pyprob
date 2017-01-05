(ns queries.captcha-wikipedia
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer [defquery with-primitive-procedures]]
            anglican.csis.csis
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-wikipedia :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
            [clojure.core.matrix :as m]))

;; QUERY
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

;; COMPILE stuff
(defn COMPILE-combine-observes-fn [observes]
  (:value (first observes)))

(def COMPILE-query-args [nil])
