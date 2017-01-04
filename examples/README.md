# Examples of Compiled Inference

## Compilation
Run with the following flags:
```
lein run -- -m compile -n queries.minimal -q minimal -o COMPILE-combine-observes-fn -a COMPILE-query-args
```

Then run the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th compile.lua --batchSize 16 --validSize 16 --validInterval 256 --obsEmbDim 16 --lstmDim 16
```

## Inference
Run with the following flags:
```
lein run -- -m infer -n queries.minimal -q minimal -Z "[[1 2]]"
```

This must be run while running the following from [torch-csis](https://github.com/tuananhle7/torch-csis):
```
th infer.lua --latest
```
