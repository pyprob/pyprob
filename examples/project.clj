(defproject examples "0.1.0-SNAPSHOT"
  :description "Inference Compilation: Examples"
  :plugins [[lein-gorilla "0.4.0"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/tools.cli "0.3.5"]
                 [anglican "1.0.0"]
                 [anglican-infcomp "0.1.0-SNAPSHOT"]
                 [net.polyc0l0r/clj-hdf5 "0.2.2-SNAPSHOT"]]
  :java-source-paths ["src/helpers"]
  :main examples.core)
