# Reproduction-of-the-Paper
**对一些论文进行复现**   
**created by nil**
***
## 文档结构
* speckle reduced  
  对应的文章的主要内容是：使用arss+sfft去进行一个散斑抑制的操作

* 


***
## 版本更新
>2024.9.28 更新  
* 基本完成了论文“Chang 等 - 2017 - Speckle reduced lensless holographic projection from phase-only computer-generated hologram”的阅读，后续还需要增加一些方法比较的内容

* 完成了代码第一版（学长给的）的运行，后续还需要增加一个s—fft的过程

* 把prop_dist 改成-0.9，保证和收敛光的距离一致
  
>2024.9.29 更新
* 完成了第一版 ARSS_SFFT的代码纂写实验截图如下，loss不知道为啥比较高，但是ssim有提升
* 后续准备增加在虚拟平面上增加矩形孔，来验证论文里的结果,并多做几个图的结果
  
![\<img alt="结果1" src="/speckle reduced/result/image0.png" ztype="zimage">](/speckle%20reduced/result/image0.png)