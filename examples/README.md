# Examples of Compiled Inference

## Minimal
### Compilation
Run:
```
lein run -- -m compile -n queries.minimal -q minimal -o COMPILE-combine-observes-fn -a COMPILE-query-args
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmbDim 16 --lstmDim 16
```

### Inference
Run:
```
lein run -- -m infer -n queries.minimal -q minimal -Z "[[1 2]]"
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## Gaussian Mixture Model with fixed number of clusters
### Compilation
Run:
```
lein run -- \
-m compile \
-n queries.gmm-fixed-number-of-clusters \
-q gmm-fixed-number-of-clusters \
-o COMPILE-combine-observes-fn \
-s COMPILE-combine-samples-fn \
-a COMPILE-query-args
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
-E INFER-observe-embedder-input \
-A INFER-query-args
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```

## Wikipedia's Captcha
### Compilation
Run:
```
lein run -- \
-m compile \
-n queries.captcha-wikipedia \
-q captcha-wikipedia \
-o COMPILE-combine-observes-fn \
-a COMPILE-query-args
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
-n queries.gmm-fixed-number-of-clusters \
-q gmm-fixed-number-of-clusters \
-E INFER-observe-embedder-input \
-A INFER-query-args
```

At the same time, run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```
