对于SVNN的拙劣模仿，本来想水篇会议，导师不让说我抄袭。
能运行出来，但是从信号上看，不是对旁瓣的波形进行抑制，而是对旁瓣所在列进行了整体的抑制。
unet也好cnn也好，总感觉学习不到这种波形上的变换，单纯的做成了识别加整体抑制、这样导致了测试图中，背景也被抑制了。
不会继续研究了。或许加入注意力机制可以改善背景被抑制？
损失函数可能也值得研究。
在训练代码的时候使用前进行了归一化-1~1，但是进行测试的时候，未进行归一化也可以出结果？这是为何？
原始论文中的仿真数据生成方法还没有搞懂或许会有帮助。
以下为部分理解：
SVA算法的设计基于奈奎斯特采样率条件。当实际应用中的采样率偏离该条件时，其旁瓣抑制性能会显著下降。在SAR图像处理过程中，距离方向的采样间隔通常较小，信号更可能满足或接近奈奎斯特采样率。
因此，SVA算法在距离方向上表现出相对稳定的旁瓣抑制性能，有效突出弱目标。

相比之下，方位分辨率受天线合成孔径长度和平台运动轨迹的影响。运动误差和采样率的变化可能导致欠采样或过采样，从而导致SVA算法在方位方向上的抑制性能显著下降。
这个问题在具有非均匀采样或运动误差的复杂场景中尤为突出。

基于上述分析，本文提出了以下改进：
• 消除SVA算法对奈奎斯特采样率的严格依赖，以增强其在不同采样条件下的鲁棒性。
• 提高方位方向的旁瓣抑制性能，克服现有方法中的性能瓶颈。

说人话就是让网络学习距离向的抑制效果，然后套到方向位上面去。提供的论文网络用的unet、但是代码用的cnn。你可以自己试试，效果不咋好。可能是方法没找对。

最后还是非常感谢Suo Yuxi博士的论文与代码：https://github.com/suoyuxi/SVNN
