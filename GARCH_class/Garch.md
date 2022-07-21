# ARCH的相关性质
## 底层由来  
从AR(p)开始：$x_t = \phi_0 + \phi_1x_{t-1}+ \phi_2x_{t-2} + ... + \phi_qx_{t-q}$ 

定义ARCH(q)模型为：$\varepsilon_t = \sigma_t\mu_t$  
where $\mu_t$ ~ iid, $\sigma_t^2 = a_0 + a_1\varepsilon_{t-1} + a_2\varepsilon_{t-2} + ... + a_q\varepsilon_{t-q}$  

## 期望、方差和协方差
对于白噪声序列{$\varepsilon_t$}, 在假定条件与方差不变的情况下，根据***条件期望的期望就是它本身***的性质，我们有  
$E\left(\varepsilon_{t}\right)=E\left[E\left(\varepsilon_{t} \mid F^{t-1}\right)\right]=0$  

$\operatorname{Var}\left(\varepsilon_{t}\right)=E\left[E\left(\varepsilon_{t}^{2} \mid F^{t-1}\right)\right]=E\left(\sigma_{t}^{2}\right)=a_{0}+a_{1} E\left(\varepsilon_{t-1}^{2}\right)+\ldots+a_{q} E\left(\varepsilon_{t-q}^{2}\right)=\frac{a_{0}}{1-a_{1}-\ldots-a_{q}}$  

$\operatorname{Cov}\left(\varepsilon_{t}, \varepsilon_{t-j}\right)=E\left(\left(\varepsilon_{t}-E\left(\varepsilon_{t}\right)\right)\left(\varepsilon_{t-j}-E\left(\varepsilon_{t-j}\right)\right)\right)=E\left(\varepsilon_{t} \varepsilon_{t-j}\right)=0$  

## 参数估计过程
1. 估计一个ARCH模型，首先需要确定好AR(p)模型的阶数，可以根据相关定阶模型。但对于波动率阶数 q 的确定，***我们要先检验序列{$\varepsilon_t$} 确实存在显著的ARCH效应，然后根据偏自相关函数(Pacf)来确定 q*** ，
2. 定好阶后，为了估计模型参数，当{$\mu_t$}服从正态分布时，我们可以用***最小二乘或最大似然估计对模型参数进行估计。***
## 建模过程（由于此模型前期工作跟ARMA模型的建立很相似，所以有些步骤将简述）
