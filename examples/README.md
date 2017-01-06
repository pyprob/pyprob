# Examples of Compiled Inference

## Jump to experiment
1. [Minimal](#1-minimal)
2. [Gaussian Mixture Model with fixed number of clusters](#2-gaussian-mixture-model-with-fixed-number-of-clusters)
3. [Wikipedia's Captcha](#3-wikipedias-captcha)
4. [Facebook's Captcha](#4-facebooks-captcha)

## 1. Minimal
### Compilation
Run:
```
lein run -- \
-m compile \
-n queries.minimal \
-q minimal \
-o COMPILE-combine-observes-fn \
-x "[[1 2]]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmbDim 16 --lstmDim 16
```

### Inference
Run:
```
lein run -- \
-m infer \
-n queries.minimal \
-q minimal \
-Z "[[1 2]]"
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
-m compile \
-n queries.gmm-fixed-number-of-clusters \
-q gmm-fixed-number-of-clusters \
-o COMPILE-combine-observes-fn \
-s COMPILE-combine-samples-fn \
-x "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmb lenet --obsEmbDim 8 --lstmDim 4 --obsSmooth
```

### Inference
Run:
```
lein run -- \
-m infer \
-n queries.gmm-fixed-number-of-clusters \
-q gmm-fixed-number-of-clusters \
-Y "$(python src/helpers/io/csv2hst.py resources/gmm-data/gmm.csv)" \
-Z "[$(python src/helpers/io/csv2edn.py resources/gmm-data/gmm.csv) {:mu-0 [0 0] :Sigma-0 [[0.1 0] [0 0.1]]}]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 3. Wikipedia's Captcha
### Compilation
Run:
```
lein run -- \
-m compile \
-n queries.captcha-wikipedia \
-q captcha-wikipedia \
-o COMPILE-combine-observes-fn \
-x "[nil]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4
```

### Inference
Run:
```
lein run -- \
-m infer \
-n queries.captcha-wikipedia \
-q captcha-wikipedia \
-Z "[$(python src/helpers/io/png2edn.py resources/wikipedia-dataset/agavelooms.png)]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## 4. Facebook's Captcha
### Compilation
Run:
```
lein run -- \
-m compile \
-n queries.captcha-facebook \
-q captcha-facebook \
-o COMPILE-combine-observes-fn \
-x "[nil]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4
```

### Inference
Run:
```
lein run -- \
-m infer \
-n queries.captcha-facebook \
-q captcha-facebook \
-Z "[$(python src/helpers/io/png2edn.py resources/facebook-dataset/2MsLet.png)]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```
