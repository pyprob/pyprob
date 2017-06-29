;; gorilla-repl.fileformat = 1

;; **
;;; # Wikipedia's Captcha
;;; 
;;; In this worksheet, we demonstrate inference compilation on a Captcha rendering generative model and use it to break Wikipedia's Captcha. More in [the paper](https://arxiv.org/abs/1610.09900).
;; **

;; @@
(ns worksheets.captcha-wikipedia
  (:require [anglican.runtime :refer :all]
            [anglican.emit :refer :all]
            [anglican.stat :as stat]
            [anglican.infcomp.zmq :as zmq]
            [anglican.inference :refer [infer]]
            [anglican.infcomp.prior :as prior]
            [gorilla-plot.core :as plt]
            [helpers.captcha :refer [levenshtein-normalized]]
            [helpers.captcha-wikipedia :refer [render render-to-file abc-dist abc-sigma letter-dict oxCaptcha]]
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
;;; {"type":"html","content":"<span class='clj-unkown'>#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--27296$fn--27297]</span>","value":"#function[anglican.infcomp.core/reset-infcomp-addressing-scheme!$fn--27296$fn--27297]"}
;; <=

;; **
;;; ## Captcha generative model
;; **

;; @@
;; CAPTCHA Query
(with-primitive-procedures [render abc-dist repeatedly]
  (defquery captcha-wikipedia [baseline-image]
    (let [;; Number of letters in CAPTCHA
          num-letters (sample (uniform-discrete 8 11))
          font-size (sample (uniform-discrete 26 32))
          kerning (sample (uniform-discrete 1 3))
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
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-wikipedia/captcha-wikipedia</span>","value":"#'worksheets.captcha-wikipedia/captcha-wikipedia"}
;; <=

;; **
;;; ## Train a compilation artifact
;; **

;; @@
(defn combine-observes-fn [observes]
  (:value (first observes)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-wikipedia/combine-observes-fn</span>","value":"#'worksheets.captcha-wikipedia/combine-observes-fn"}
;; <=

;; @@
(def replier (zmq/start-replier captcha-wikipedia [nil] combine-observes-fn))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-wikipedia/replier</span>","value":"#'worksheets.captcha-wikipedia/replier"}
;; <=

;; @@
(zmq/stop-replier replier)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-string'>&quot;ZMQ connection terminated.&quot;</span>","value":"\"ZMQ connection terminated.\""}
;; <=

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
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-wikipedia/ground-truth-letters</span>","value":"#'worksheets.captcha-wikipedia/ground-truth-letters"}
;; <=

;; **
;;; ### Load synthetic Wikipedia Captchas
;; **

;; @@
(def num-observes 100)
(def samples-from-prior (take num-observes (prior/sample-from-prior captcha-wikipedia nil)))
(def observes (map (comp combine-observes-fn :observes) samples-from-prior))
(def ground-truth-letters (map (fn [smp]
                                 (let [latents (:samples smp)
                                       letter-ids (map :value (filter #(= (:sample-address %) "letterid") latents))
                                       letters (apply str (map (partial nth letter-dict) letter-ids))]
                                   letters))
                               samples-from-prior))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;worksheets.captcha-wikipedia/ground-truth-letters</span>","value":"#'worksheets.captcha-wikipedia/ground-truth-letters"}
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
                            (take num-particles (infer :smc captcha-wikipedia [observe] :number-of-particles num-particles)))
                          observes))
(def smc-MAP-list (map (comp empirical-MAP stat/collect-results) smc-states-list))
(time
  (doall (map (fn [smc-MAP filename] (render-to-file (:letters smc-MAP) (:font-size smc-MAP) (:kerning smc-MAP) filename))
              smc-MAP-list
              (map #(str "plots/captcha-wikipedia/" % "-smc.png") (range 1 (inc (count observes)))))))

;; RMH
(def num-iters 1)
(def rmh-states-list (map (fn [observe]
                            (take num-iters (infer :rmh captcha-wikipedia [observe])))
                          observes))
(def rmh-posterior-list (map (comp first last stat/collect-results) rmh-states-list))
(time
  (doall (map (fn [rmh-posterior filename] (render-to-file (:letters rmh-posterior) (:font-size rmh-posterior) (:kerning rmh-posterior) filename))
              rmh-posterior-list
              (map #(str "plots/captcha-wikipedia/" % "-rmh.png") (range 1 (inc (count observes)))))))

;; CSIS
(def num-particles 1)
(def csis-states-list (map (fn [observe]
                             (take num-particles (infer :csis captcha-wikipedia [observe])))
                           observes))
(def csis-MAP-list (map (comp empirical-MAP stat/collect-results) csis-states-list))
(time
  (doall (map (fn [csis-MAP filename] (render-to-file (:letters csis-MAP) (:font-size csis-MAP) (:kerning csis-MAP) filename))
              csis-MAP-list
              (map #(str "plots/captcha-wikipedia/" % "-csis.png") (range 1 (inc (count observes)))))))
;; @@
;; ->
;;; &quot;Elapsed time: 4465.442 msecs&quot;
;;; &quot;Elapsed time: 6990.577 msecs&quot;
;;; &quot;Elapsed time: 9159.338 msecs&quot;
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
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-string'>&quot;gbbnxpbb&quot;</span>","value":"\"gbbnxpbb\""},{"type":"html","content":"<span class='clj-string'>&quot;pgzbgggg&quot;</span>","value":"\"pgzbgggg\""},{"type":"html","content":"<span class='clj-string'>&quot;pphagppp&quot;</span>","value":"\"pphagppp\""},{"type":"html","content":"<span class='clj-string'>&quot;bgqgbnbbgx&quot;</span>","value":"\"bgqgbnbbgx\""},{"type":"html","content":"<span class='clj-string'>&quot;pphppppp&quot;</span>","value":"\"pphppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;ppplppgp&quot;</span>","value":"\"ppplppgp\""},{"type":"html","content":"<span class='clj-string'>&quot;gbggggpbbp&quot;</span>","value":"\"gbggggpbbp\""},{"type":"html","content":"<span class='clj-string'>&quot;gpgpgpbgxg&quot;</span>","value":"\"gpgpgpbgxg\""},{"type":"html","content":"<span class='clj-string'>&quot;ggipgppp&quot;</span>","value":"\"ggipgppp\""},{"type":"html","content":"<span class='clj-string'>&quot;pxpgpsxr&quot;</span>","value":"\"pxpgpsxr\""},{"type":"html","content":"<span class='clj-string'>&quot;pppagpgp&quot;</span>","value":"\"pppagpgp\""},{"type":"html","content":"<span class='clj-string'>&quot;lldrlpbh&quot;</span>","value":"\"lldrlpbh\""},{"type":"html","content":"<span class='clj-string'>&quot;ppgppppp&quot;</span>","value":"\"ppgppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;bpgbbhbbb&quot;</span>","value":"\"bpgbbhbbb\""},{"type":"html","content":"<span class='clj-string'>&quot;pbgggggw&quot;</span>","value":"\"pbgggggw\""},{"type":"html","content":"<span class='clj-string'>&quot;ppppppppp&quot;</span>","value":"\"ppppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;bcbzhbbb&quot;</span>","value":"\"bcbzhbbb\""},{"type":"html","content":"<span class='clj-string'>&quot;hhphspph&quot;</span>","value":"\"hhphspph\""},{"type":"html","content":"<span class='clj-string'>&quot;gggsgggg&quot;</span>","value":"\"gggsgggg\""},{"type":"html","content":"<span class='clj-string'>&quot;pppphppp&quot;</span>","value":"\"pppphppp\""},{"type":"html","content":"<span class='clj-string'>&quot;pppepphg&quot;</span>","value":"\"pppepphg\""},{"type":"html","content":"<span class='clj-string'>&quot;pgpbpppp&quot;</span>","value":"\"pgpbpppp\""},{"type":"html","content":"<span class='clj-string'>&quot;seshggbhz&quot;</span>","value":"\"seshggbhz\""},{"type":"html","content":"<span class='clj-string'>&quot;bktgxbteb&quot;</span>","value":"\"bktgxbteb\""},{"type":"html","content":"<span class='clj-string'>&quot;pqplxxuxs&quot;</span>","value":"\"pqplxxuxs\""},{"type":"html","content":"<span class='clj-string'>&quot;bgphhfhgg&quot;</span>","value":"\"bgphhfhgg\""},{"type":"html","content":"<span class='clj-string'>&quot;pppxppprp&quot;</span>","value":"\"pppxppprp\""},{"type":"html","content":"<span class='clj-string'>&quot;hzbpshhzh&quot;</span>","value":"\"hzbpshhzh\""},{"type":"html","content":"<span class='clj-string'>&quot;pppppppp&quot;</span>","value":"\"pppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;ffpgigwg&quot;</span>","value":"\"ffpgigwg\""},{"type":"html","content":"<span class='clj-string'>&quot;gppgbgpp&quot;</span>","value":"\"gppgbgpp\""},{"type":"html","content":"<span class='clj-string'>&quot;hhbhehff&quot;</span>","value":"\"hhbhehff\""},{"type":"html","content":"<span class='clj-string'>&quot;bbbpbebb&quot;</span>","value":"\"bbbpbebb\""},{"type":"html","content":"<span class='clj-string'>&quot;wggggggg&quot;</span>","value":"\"wggggggg\""},{"type":"html","content":"<span class='clj-string'>&quot;pgppppppp&quot;</span>","value":"\"pgppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;pspppppp&quot;</span>","value":"\"pspppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;iiggggidg&quot;</span>","value":"\"iiggggidg\""},{"type":"html","content":"<span class='clj-string'>&quot;pphnppppap&quot;</span>","value":"\"pphnppppap\""},{"type":"html","content":"<span class='clj-string'>&quot;ezggfbcc&quot;</span>","value":"\"ezggfbcc\""},{"type":"html","content":"<span class='clj-string'>&quot;pxppppppp&quot;</span>","value":"\"pxppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;hcphxchx&quot;</span>","value":"\"hcphxchx\""},{"type":"html","content":"<span class='clj-string'>&quot;bpppgwppp&quot;</span>","value":"\"bpppgwppp\""},{"type":"html","content":"<span class='clj-string'>&quot;bkbpngbbp&quot;</span>","value":"\"bkbpngbbp\""},{"type":"html","content":"<span class='clj-string'>&quot;ppppppplpp&quot;</span>","value":"\"ppppppplpp\""},{"type":"html","content":"<span class='clj-string'>&quot;bbbggbggb&quot;</span>","value":"\"bbbggbggb\""},{"type":"html","content":"<span class='clj-string'>&quot;npgptggh&quot;</span>","value":"\"npgptggh\""},{"type":"html","content":"<span class='clj-string'>&quot;hprgxggggh&quot;</span>","value":"\"hprgxggggh\""},{"type":"html","content":"<span class='clj-string'>&quot;gbbbbxxx&quot;</span>","value":"\"gbbbbxxx\""},{"type":"html","content":"<span class='clj-string'>&quot;bpffffffa&quot;</span>","value":"\"bpffffffa\""},{"type":"html","content":"<span class='clj-string'>&quot;cxccccchc&quot;</span>","value":"\"cxccccchc\""},{"type":"html","content":"<span class='clj-string'>&quot;ipppppgpp&quot;</span>","value":"\"ipppppgpp\""},{"type":"html","content":"<span class='clj-string'>&quot;pplppppp&quot;</span>","value":"\"pplppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;perzpzgg&quot;</span>","value":"\"perzpzgg\""},{"type":"html","content":"<span class='clj-string'>&quot;gbbbbbbg&quot;</span>","value":"\"gbbbbbbg\""},{"type":"html","content":"<span class='clj-string'>&quot;tbxhcccr&quot;</span>","value":"\"tbxhcccr\""},{"type":"html","content":"<span class='clj-string'>&quot;pppppppp&quot;</span>","value":"\"pppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;pppppppcpp&quot;</span>","value":"\"pppppppcpp\""},{"type":"html","content":"<span class='clj-string'>&quot;bbpbbpggpb&quot;</span>","value":"\"bbpbbpggpb\""},{"type":"html","content":"<span class='clj-string'>&quot;ggcchbgbh&quot;</span>","value":"\"ggcchbgbh\""},{"type":"html","content":"<span class='clj-string'>&quot;ppqnppdh&quot;</span>","value":"\"ppqnppdh\""},{"type":"html","content":"<span class='clj-string'>&quot;bnxphbpp&quot;</span>","value":"\"bnxphbpp\""},{"type":"html","content":"<span class='clj-string'>&quot;bdbezdbi&quot;</span>","value":"\"bdbezdbi\""},{"type":"html","content":"<span class='clj-string'>&quot;rdrhrprrp&quot;</span>","value":"\"rdrhrprrp\""},{"type":"html","content":"<span class='clj-string'>&quot;shanslsh&quot;</span>","value":"\"shanslsh\""},{"type":"html","content":"<span class='clj-string'>&quot;pbdbpbbp&quot;</span>","value":"\"pbdbpbbp\""},{"type":"html","content":"<span class='clj-string'>&quot;gcsbcmyl&quot;</span>","value":"\"gcsbcmyl\""},{"type":"html","content":"<span class='clj-string'>&quot;bbbbbbxbp&quot;</span>","value":"\"bbbbbbxbp\""},{"type":"html","content":"<span class='clj-string'>&quot;ggptggcp&quot;</span>","value":"\"ggptggcp\""},{"type":"html","content":"<span class='clj-string'>&quot;ggcggggegg&quot;</span>","value":"\"ggcggggegg\""},{"type":"html","content":"<span class='clj-string'>&quot;kppbggzgpg&quot;</span>","value":"\"kppbggzgpg\""},{"type":"html","content":"<span class='clj-string'>&quot;papbgbabg&quot;</span>","value":"\"papbgbabg\""},{"type":"html","content":"<span class='clj-string'>&quot;bgcgbglg&quot;</span>","value":"\"bgcgbglg\""},{"type":"html","content":"<span class='clj-string'>&quot;hbixbpbp&quot;</span>","value":"\"hbixbpbp\""},{"type":"html","content":"<span class='clj-string'>&quot;hpdpfpcpp&quot;</span>","value":"\"hpdpfpcpp\""},{"type":"html","content":"<span class='clj-string'>&quot;pggggpbgbp&quot;</span>","value":"\"pggggpbgbp\""},{"type":"html","content":"<span class='clj-string'>&quot;nnnxpnbq&quot;</span>","value":"\"nnnxpnbq\""},{"type":"html","content":"<span class='clj-string'>&quot;phpppici&quot;</span>","value":"\"phpppici\""},{"type":"html","content":"<span class='clj-string'>&quot;pbppgbtg&quot;</span>","value":"\"pbppgbtg\""},{"type":"html","content":"<span class='clj-string'>&quot;plpggbpg&quot;</span>","value":"\"plpggbpg\""},{"type":"html","content":"<span class='clj-string'>&quot;ggggggggc&quot;</span>","value":"\"ggggggggc\""},{"type":"html","content":"<span class='clj-string'>&quot;ppcpxcsnp&quot;</span>","value":"\"ppcpxcsnp\""},{"type":"html","content":"<span class='clj-string'>&quot;ppppbpppp&quot;</span>","value":"\"ppppbpppp\""},{"type":"html","content":"<span class='clj-string'>&quot;ppppxxppx&quot;</span>","value":"\"ppppxxppx\""},{"type":"html","content":"<span class='clj-string'>&quot;fpppppep&quot;</span>","value":"\"fpppppep\""},{"type":"html","content":"<span class='clj-string'>&quot;ggzggnqg&quot;</span>","value":"\"ggzggnqg\""},{"type":"html","content":"<span class='clj-string'>&quot;pbbgmpgg&quot;</span>","value":"\"pbbgmpgg\""},{"type":"html","content":"<span class='clj-string'>&quot;ccigipip&quot;</span>","value":"\"ccigipip\""},{"type":"html","content":"<span class='clj-string'>&quot;pppbbplpb&quot;</span>","value":"\"pppbbplpb\""},{"type":"html","content":"<span class='clj-string'>&quot;bpqdbbgp&quot;</span>","value":"\"bpqdbbgp\""},{"type":"html","content":"<span class='clj-string'>&quot;bpepbbpg&quot;</span>","value":"\"bpepbbpg\""},{"type":"html","content":"<span class='clj-string'>&quot;bcegpnggx&quot;</span>","value":"\"bcegpnggx\""},{"type":"html","content":"<span class='clj-string'>&quot;xpplpdpz&quot;</span>","value":"\"xpplpdpz\""},{"type":"html","content":"<span class='clj-string'>&quot;xprxxxrx&quot;</span>","value":"\"xprxxxrx\""},{"type":"html","content":"<span class='clj-string'>&quot;rgbgpwsb&quot;</span>","value":"\"rgbgpwsb\""},{"type":"html","content":"<span class='clj-string'>&quot;pppppppppp&quot;</span>","value":"\"pppppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;hkgggpeg&quot;</span>","value":"\"hkgggpeg\""},{"type":"html","content":"<span class='clj-string'>&quot;rpgbwpgp&quot;</span>","value":"\"rpgbwpgp\""},{"type":"html","content":"<span class='clj-string'>&quot;pqpppppppp&quot;</span>","value":"\"pqpppppppp\""},{"type":"html","content":"<span class='clj-string'>&quot;qggdbwfg&quot;</span>","value":"\"qggdbwfg\""},{"type":"html","content":"<span class='clj-string'>&quot;ppepegxpp&quot;</span>","value":"\"ppepegxpp\""}],"value":"(\"gbbnxpbb\" \"pgzbgggg\" \"pphagppp\" \"bgqgbnbbgx\" \"pphppppp\" \"ppplppgp\" \"gbggggpbbp\" \"gpgpgpbgxg\" \"ggipgppp\" \"pxpgpsxr\" \"pppagpgp\" \"lldrlpbh\" \"ppgppppp\" \"bpgbbhbbb\" \"pbgggggw\" \"ppppppppp\" \"bcbzhbbb\" \"hhphspph\" \"gggsgggg\" \"pppphppp\" \"pppepphg\" \"pgpbpppp\" \"seshggbhz\" \"bktgxbteb\" \"pqplxxuxs\" \"bgphhfhgg\" \"pppxppprp\" \"hzbpshhzh\" \"pppppppp\" \"ffpgigwg\" \"gppgbgpp\" \"hhbhehff\" \"bbbpbebb\" \"wggggggg\" \"pgppppppp\" \"pspppppp\" \"iiggggidg\" \"pphnppppap\" \"ezggfbcc\" \"pxppppppp\" \"hcphxchx\" \"bpppgwppp\" \"bkbpngbbp\" \"ppppppplpp\" \"bbbggbggb\" \"npgptggh\" \"hprgxggggh\" \"gbbbbxxx\" \"bpffffffa\" \"cxccccchc\" \"ipppppgpp\" \"pplppppp\" \"perzpzgg\" \"gbbbbbbg\" \"tbxhcccr\" \"pppppppp\" \"pppppppcpp\" \"bbpbbpggpb\" \"ggcchbgbh\" \"ppqnppdh\" \"bnxphbpp\" \"bdbezdbi\" \"rdrhrprrp\" \"shanslsh\" \"pbdbpbbp\" \"gcsbcmyl\" \"bbbbbbxbp\" \"ggptggcp\" \"ggcggggegg\" \"kppbggzgpg\" \"papbgbabg\" \"bgcgbglg\" \"hbixbpbp\" \"hpdpfpcpp\" \"pggggpbgbp\" \"nnnxpnbq\" \"phpppici\" \"pbppgbtg\" \"plpggbpg\" \"ggggggggc\" \"ppcpxcsnp\" \"ppppbpppp\" \"ppppxxppx\" \"fpppppep\" \"ggzggnqg\" \"pbbgmpgg\" \"ccigipip\" \"pppbbplpb\" \"bpqdbbgp\" \"bpepbbpg\" \"bcegpnggx\" \"xpplpdpz\" \"xprxxxrx\" \"rgbgpwsb\" \"pppppppppp\" \"hkgggpeg\" \"rpgbwpgp\" \"pqpppppppp\" \"qggdbwfg\" \"ppepegxpp\")"}
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
;;; SMC :  0.9335833305120468 
;;; RMH :  0.9246388864517212 
;;; CSIS:  0.9334444415569305
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
