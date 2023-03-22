根据三分类问题的交叉熵损失公式，有：

$$
\mathcal{L}=-\sum_{i=0}^2 y_i\log(\hat{y}_i)
$$

其中，$y_i$表示真实标签的one-hot编码，$\hat{y}_i$表示模型对类别$i$的预测概率。因为只有一个样本，可以直接计算出该样本的预测概率向量：

$$
\begin{bmatrix}\hat{y}_0 & \hat{y}_1 & \hat{y}_2\end{bmatrix} = \begin{bmatrix} 0.1 & 0.1 & 0.2 \\ 0.2 & 0.1 & 0.2 \end{bmatrix} \begin{bmatrix} 0.2 \\ 0.3 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.05 \\ 0.14 \\ 0.08 \end{bmatrix}
$$

由于真实标签为$i$，因此$y_i=1$，$y_j=0$（$j\neq i$）。代入公式，得到交叉熵损失为：

$$
\begin{aligned}
\mathcal{L} &= -\sum_{i=0}^2 y_i\log(\hat{y}_i) \\
&= -\left(1\times\log(\hat{y}_i)+0\times\log(\hat{y}_j)+0\times\log(\hat{y}_k)\right) \\
&= -\log(\hat{y}_i) \\
&= -\log(\hat{y}_0) \\
&= -\log(0.05) \\
&\approx 2.996
\end{aligned}
$$

因此，该样本的交叉熵损失约为2.996。
