# 逻辑斯特回归
使用sgd优化

- 公式推导: 参考此[csdn博客](http://blog.csdn.net/xiaoxiangzi222/article/details/55097570)

- 编译运行： `make&&./main`

- 结果：
```
# loading traindata...
# training examples: 2000
# features:          7034
# loading testdata...
# testing examples: 600
# sgd begining~
# convergence: 0.0917 iterations: 20
# convergence: 0.0759 iterations: 40
# convergence: 0.0663 iterations: 60
# convergence: 0.0598 iterations: 80
# convergence: 0.0550 iterations: 100
# convergence: 0.0511 iterations: 120
# convergence: 0.0479 iterations: 140
# convergence: 0.0451 iterations: 160
# convergence: 0.0426 iterations: 180
# convergence: 0.0404 iterations: 200
# accuracy:    0.9117 (547/600)
# precision:   0.9103
# recall:      0.9133
# tp:          274
# tn:          273
# fp:          27
# fn:          26

```
