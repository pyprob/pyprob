# Examples of Inference Compilation and Universal Probabilistic Programming

## Contents
1. [Minimal](#1-minimal)
2. [Gaussian Mixture Model with fixed number of clusters](#2-gaussian-mixture-model-with-fixed-number-of-clusters)
3. [Gaussian Mixture Model with variable number of clusters](#3-gaussian-mixture-model-with-variable-number-of-clusters)
4. [Wikipedia's Captcha](#4-wikipedias-captcha)
5. [Facebook's Captcha](#5-facebooks-captcha)

## 1. Minimal
### Compilation
Run:
```
lein run -- \
--mode compile \
--namespace queries.minimal \
--query minimal \
--compile-combine-observes-fn combine-observes-fn \
--compile-query-args-value "[[1 2]]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmbDim 16 --lstmDim 16
```

### Inference
Run:
```
lein run -- \
--mode infer \
--namespace queries.minimal \
--query minimal \
--infer-query-args-value "[[1 2]]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 2. Gaussian Mixture Model with fixed number of clusters
### Compilation
Run:
```
lein run -- \
--mode compile \
--namespace queries.gmm-fixed-number-of-clusters \
--query gmm-fixed-number-of-clusters \
--compile-combine-observes-fn combine-observes-fn \
--compile-combine-samples-fn combine-samples-fn \
--compile-query-args-value "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmb lenet --obsEmbDim 8 --lstmDim 4 --obsSmooth
```

### Inference
Run:
```
lein run -- \
--mode infer \
--namespace queries.gmm-fixed-number-of-clusters \
--query gmm-fixed-number-of-clusters \
--infer-observe-embedder-input-value "$(python src/helpers/io/csv2hst.py resources/gmm-data/gmm.csv)" \
--infer-query-args-value "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 3. Gaussian Mixture Model with variable number of clusters
### Compilation
Run:
```
lein run -- \
--mode compile \
--namespace queries.gmm-variable-number-of-clusters \
--query gmm-variable-number-of-clusters \
--compile-combine-observes-fn combine-observes-fn \
--compile-combine-samples-fn combine-samples-fn \
--compile-query-args-value "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmb lenet --obsEmbDim 8 --lstmDim 4 --obsSmooth
```

### Inference
Run:
```
lein run -- \
--mode infer \
--namespace queries.gmm-variable-number-of-clusters \
--query gmm-variable-number-of-clusters \
--infer-observe-embedder-input-value "$(python src/helpers/io/csv2hst.py resources/gmm-data/gmm.csv)" \
--infer-query-args-value "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 4. Wikipedia's Captcha
### Compilation
Run:
```
lein run -- \
--mode compile \
--namespace queries.captcha-wikipedia \
--query captcha-wikipedia \
--compile-combine-observes-fn combine-observes-fn \
--compile-query-args-value "[nil]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4
```

### Inference
Run:
```
lein run -- \
--mode infer \
--namespace queries.captcha-wikipedia \
--query captcha-wikipedia \
--infer-query-args-value "[$(python src/helpers/io/png2edn.py resources/wikipedia-dataset/agavelooms.png)]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 5. Facebook's Captcha
### Compilation
Run:
```
lein run -- \
--mode compile \
--namespace queries.captcha-facebook \
--query captcha-facebook \
--compile-combine-observes-fn combine-observes-fn \
--compile-query-args-value "[nil]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4
```

### Inference
Run:
```
lein run -- \
--mode infer \
--namespace queries.captcha-facebook \
--query captcha-facebook \
--infer-query-args-value "[$(python src/helpers/io/png2edn.py resources/facebook-dataset/2MsLet.png)]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```
