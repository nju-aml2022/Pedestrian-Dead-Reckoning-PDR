<style>
h1 {
    text-align: center;
}
h2, h3 {
    page-break-after: avoid; 
}
.center {
    margin: 0 auto;
    width: fit-content;
    margin-top: 2em;
    padding-top: 0.5em;
    padding-bottom: 0.5em;
    margin-bottom: 2em;
}
.title {
    font-weight: bold;
    border-top-style: solid;
    border-bottom-style: solid;
}
.newpage {
    page-break-after: always
}
@media print {
    @page {
        margin: 3cm;
    }
}
</style>

<h1 style="margin-top: 4em">
高级机器学习作业
</h1>

# <h1 class="center title">PDR: 室内行人移动方位推算技术</h1>

<div class="center">
<h3>院系：人工智能学院</h3>
<h3>组长：方盛俊 (201300035)</h3>
<h3>组员：任圣杰 (201300036)<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;曹明隽 (201300031)</h3>
<h3>邮箱：201300035@smail.nju.edu.cn</h3>
<h3>时间：2022 年 11 月 30 日</h3>
</div>

<div class="newpage"></div>

<!-- 生成目录 -->

## <h1>目录</h1>

[TOC]

<div class="newpage"></div>

<!-- 文章主体内容 -->

## 一、问题描述

### 1.1 问题背景

室内 GPS 信号受到阻隔，尤其时车库场景无线接入点 AP 较为稀疏，室内定位很难通过 GPS 或位置指纹等技术实现。行人行位推算技术（Pedestrian Dead Reckoning，PDR）可以有效解决这一问题，它利用手机内置传感器收集数据，通过轻量级低功耗算法运行框架和技术，实现稳定可靠、高精度的方位推算计算。

### 1.2  输入

输入为 CSV 格式，其每一行都有数据值及采集到该数据时的时间（从开始收集计算），以加速度计为例：

```csv
"Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)",
"Acceleration z (m/s^2)"
9.364549000E-3,1.003170490E0,3.543418407E0,7.366958618E0
1.136538300E-2,9.696516991E-1,3.600879431E0,7.357381821E0
1.336699700E-2,9.696516991E-1,3.620033026E0,7.343016624E0
```

输入数据包括：

1. `Accelerometer.csv, Linear Accelerometer.csv`：50Hz 采集的加速度和线加速度，单位为 $m/s^2$
2. `Magnetometer.csv`：50Hz 采集的磁力计读数，单位为 $\mu\text{T}$
3. `Gyroscope.csv`：50Hz 采集的陀螺仪读数，单位为 $\text{rad}/s$
4. `Location_input.csv`：前 10% 时间的 1Hz 的 GPS 位置读数：
   1. 经纬度 `Latitude, Longitude`，单位为 $\degree$
   2. 高度 `Height`，单位为 $m$
   3. 速度 `Velocity`，单位为 $m/s$
   4. 方位角 `Direction`，单位为 $\degree$
   5. 垂直和水平精度 `Horizontal Accuracy, Vertical Accuracy`，单位为 $m$

其中加速度、线加速度、磁力计、陀螺仪均有三个维度上的读数：

<div style="text-align: center;">
    <img alt="" src="images/2022-12-01-15-09-28.png" width="50%" style="margin: 0 auto;" />
</div>
<div style="text-align: center; font-size: small">
    <b>Figure 1.</b> 数据的三个维度
    <br />
    <br />
</div>


### 1.3 输出

输出为后 90% 时间的 1Hz 的 GPS 位置读数中的经纬度和方位角，即不需要输出速度、高度和精度。

输出也是 CSV 格式。输出文件 `Location_output.csv` 的前 10% 将复制 `Location_input.csv` ，并填充其后 90% 的经纬度和方位角。

**文档要求：**

- **统一使用中文标点符号；**
- **中文和英文之间要隔一个空格；**
- **每一个段落空一行。**
- **图片统一保存在 `./images` 文件夹下**


## 二、数据收集与处理

### 2.1 数据收集

数据收集工作由多组合作完成，使用软件 phyphox（中文名：手机物理工坊）。

针对室内场景，我们在多种设备上收集并使用了以下三种状态采集到的数据：

1. 步行，手机拿在手上，共49组
3. 步行，手机放在背包内，共4组
1. 步行，手机放在裤兜内，共12组

每种数据都收集了多组。在单次收集过程中，手机大部分时间保持在同一个状态下。

此外我们还收集了一些不太可能出现在室内的运动状态，用于测试模型的迁移性能：

1. 骑车，手机拿在手上，共5组
2. 骑车，手机放在背包内，共1组
3. 跑步，手机放在裤兜内，共4组
4. 骑车，手机放在裤兜内，共7组


### 2.2 数据划分

由于模型的特点，我们没有使用训练集，不需要进行数据划分，收集的所有数据都用于测试。


## 三、总体思路

### 3.1 前期思考

（方盛俊）

经过分析，我们发现 PDR 算法的关键只有两点：

1. 如何预测出任给一个时间点的已走过的 **路程**。
1. 如何预测出任给一个时间点的对应 **前进方向**。

我们很容易猜想 **路程** 在 **理论上** 应该是通过 **加速计加速度** 二重积分得到的，方向应该是经过 **陀螺仪角速度** 一重积分得到的。

但是事实上理论和实践有着很大的差距，我们测量出来的 **加速度** 和 **角速度** 总是在不断的振荡，与真实数据存在非常大的误差，这是因为：

1. 手机并不是固定在人的身上的，例如手机拿在手上时，就会随着手的活动不断摇摆振荡；
2. 手机自带的加速度计和陀螺仪就存在着一定的误差。

所以根据积分来计算出路程和方向的想法是不可行的，我们需要思考准确性和稳定性更强的方案。

这里我们发现：

1. 加速度的幅值的振荡频率和行人的步伐振荡频率强相关；
2. 磁力计的磁场强度与前进方向强相关（磁场强度方向指向北方）；
3. 我们拥有前 10% 的数据，可以从中学得很多有用的信息，比如每走一步的长度，再比如初始的前进方向。

根据这三个线索，我们制定了下面的 PDR 算法思路。


### 3.2 二级标题

### 3.1.1 三级标题

总体思路为步伐检测和方向预测。（任圣杰、方盛俊)

公式：

$$
\begin{equation}
    \begin{cases}
        X_k = X_{k-1} + L_{k-1,k} \cdot \cos(\theta_{k-1, k}) \\
        Y_k = Y_{k-1} + L_{k-1,k} \cdot \sin(\theta_{k-1, k}) \\
    \end{cases}
\end{equation}
$$

模型使用的算法没有限制，可以自行选择自己认为合理的算法。比如，可以直接把这个任务看成一个线性或者非线性的回归问题，也可以将此任务看成一个时间序列预测的问题，或者训练模型分别预测移动方向和移动速度，从而计算出当前位置，亦或利用强化学习解决这个问题等。但需要在实验报告中体现出该方法的合理性。


## 四、数据预处理

### 4.1 经纬度转换

由于输入和输出的位置信息是以经纬度为准的，而经纬度作为位置信息有一些缺点：

1. 经纬度是绝对位置，不是相对位置，后续处理比较困难；
1. 经纬度的单位长度的实际物理长度过大，比如经纬度的一度可能就对应着几十或上百公里，而我们实际的行走范围可能不会超过一公里；
2. 经纬度是在球形地球的假设上进行计算的，难以直接转换成以米为单位，也难以转到平面坐标系。

针对上面问题，我们使用了一个公式：

$$
\begin{equation}
    \begin{cases}
        X_k = K \cdot (\mathrm{latitude}_k - \mathrm{latitude\_origin}) \\
        Y_k = K \cdot (\mathrm{longitude}_k - \mathrm{longitude\_origin}) \\
    \end{cases}
\end{equation}
$$

对经纬度进行转换，其中 $K = 10^{5}$，而 $\mathrm{latitude\_origin}$ 和 $\mathrm{longitude\_origin}$ 为我们选定的初始坐标点，这里我们选用了第 10% 行数据对应的经纬度为初始坐标点。

而 $K = 10^{5}$ 是为了将数据放大到和单位米相同的数量级，由于我们的运动范围较小，因此我们可以直接忽略球形地球假设，直接使用平面坐标系。


<div style="text-align: center;">
    <img alt="" src="images/example.png" width="80%" style="margin: 0 auto;" />
</div>
<div style="text-align: center; font-size: small">
    <b>Figure 2.</b> 以 X 为横坐标，以 Y 为纵坐标得到的平面坐标系图
    <br />
    <br />
</div>

经过经纬度转换后，我们使用 $X$ 和 $Y$ 的数据画出 `test_case0` 对应平面坐标系，并且每隔一段画出对应的前进方向的箭头，我们可以看出前进方向和我们转换后的 $X$ 和 $Y$ 依旧是吻合的。

### 4.2 最近邻插值对齐数据

我们的输入有五个文件（忽略气压计数据），分别为：

- `Accelerometer.csv`（加速度计数据，50 Hz）
- `Linear Accelerometer.csv`（线性加速度计数据，50 Hz）
- `Gyroscope.csv`（陀螺仪数据，50 Hz）
- `Magnetometer.csv`（磁力计数据，50 Hz）
- `Location_input.csv`（GPS 位置信息，1 Hz）

可以看出，GPS 位置信息和其他的输入数据的频率并不相同，它们的频率 **在理论上** 是 1: 50 的关系。但是由于每个传感器的周期并不完全同步，也存在着时钟周期误差的影响，最后导致实际记录的数据实际上并不是严格 1: 50 对齐的，有可能每一个文件的数据长度都各不相同。

数据不对齐以及长度不一致的话，在数据处理和计算时会造成各种错误，甚至代码无法执行，所以我们首先一个很重要的任务就是对齐数据，让它们数据长度严格保证是 1: 50 的比例，且在时间轴上（或者说在同一个下标下）严格对齐。

在这里，我们使用了最近邻插值的方法，以 `Location_input.csv` 的时间轴为基准，进行了最近邻插值，对数据进行了对齐处理。大致算法如下：

1. 将 `Location_input.csv` 的 1 Hz 的时间轴 `time_location` 线性扩充成 50 Hz 的时间轴 `time`, 例如 `time_locaiton = [1., 2.]` 的话，便会扩充成 `time = [1., 1.02, 1.04, ..., 1.98, 2., 2.02, 2.04, ..., 2.98]`, 其中 `len(time) = 100`。
2. 将其他 50 Hz 的数据通过最近邻插值对齐到 `time` 时间轴上，最近邻插值即为每一个点匹配到最近的样本点。例如 `Accelerometer.csv` 的数据 `[[1.01, 1], [1.02, 2], [1.03, 3], [1.04, 4]]` 可能就会被最近邻插值为 `[[1.00, 1], [1.02, 2], [1.04, 4]]`。

通过最近邻插值，我们就能保证数据是严格的 1: 50 沿着时间轴对齐了。例如 `test_case` 的数据经过数据预处理后的尺寸如下：

```python
test_case.time_location.shape = (601,)
test_case.time.shape = (30050,)
```

### 4.3 对应的代码以及 TestCase 接口

最近邻插值的代码为：

```python
def nearest_neighbor_interpolation(time, time_data, data):
    '''
    使用最近邻插值获取新的 data_interp
    '''
    data_interp = []
    # 当前下标 i
    i = 0
    for t in time:
        while i < len(time_data) - 1 and t >= time_data[i + 1]:
            i += 1
        data_interp.append(data[i])
    return np.array(data_interp)
```

我们封装了一个类 `TestCase`，用于加载数据、数据预处理、数据抽象和数据保存。`TestCase` 类保存在 `dataloader.py` 文件下，部分接口如下：

```python
# 通过文件路径加载一个 TestCase 
test_case = TestCase("test_case0")
# 可以对 TestCase 进行切片, 单位为秒
new_test_case = test_case.slice(100, 500)
# 1 Hz 的时间轴
print(test_case.time_location)
# 50 Hz 的时间轴
print(test_case.time)
```

```python
# 加速度数据
print(test_case.a)
print(test_case.a_x)
print(test_case.a_y)
print(test_case.a_z)
# 加速度幅值
print(test_case.a_mag)
# 磁力计数据
print(test_case.m)
print(test_case.m_x)
print(test_case.m_y)
print(test_case.m_z)
```

```python
# 必然存在的前 10% 的数据 (Location_input.csv)
print(test_case.x)
print(test_case.y)
print(test_case.direction)
# 对应的长度
print(test_case.len_input)
# 可能存在的验证数据 (Location.csv)
if test_case.have_location_valid:
    print(test_case.x_valid)
    print(test_case.y_valid)
    print(test_case.direction_valid)
# 可能存在的输出数据 (Location_output.csv)
if test_case.have_location_output:
    print(test_case.x_output)
    print(test_case.y_output)
    print(test_case.direction_output)
```

```python
# 将结果存储到 test_case 中, 会保存成 "Location_output.csv" 文件
test_case.set_location_output(x_output, y_output, direction_output)
# 评估模型, 输出 Distances error, Direction error, Direction ratio
test_case.eval_model()
# 画出 test_case 的路线图像
test_case.draw_route()
```

并且在数据预处理后，还会输出一个对齐了的数据对应的 CSV 文件 `preprocessed.csv`，格式大致为：

```csv
t,a_x,a_y,a_z,la_x,la_y,la_z,gs_x,gs_y,gs_z,m_x,m_y,m_z
0.0,-1.2,5.2,8.2,-0.4,-0.0,-0.2,-0.2,0.4,0.1,-31.0,-22.4,-29.8
0.0,-1.2,5.2,8.2,-0.4,-0.0,-0.2,-0.2,0.4,0.1,-31.0,-22.4,-29.8
```


## 五、步伐检测

尝试使用了哪些方法，为什么使用这些方法，方法的效果怎么样，如果效果不好，可能的原因时什么，如何解决这些问题。以及在迁移场景的完成情况。（任圣杰）


## 六、方向预测

尝试使用了哪些方法，为什么使用这些方法，方法的效果怎么样，如果效果不好，可能的原因时什么，如何解决这些问题。以及迁移场景的完成情况。（方盛俊）


## 七、统一算法

如何合并以上的代码（任圣杰、方盛俊）


## 八、项目代码

### 8.1 环境搭建

如何搭建环境。（曹明隽）

### 8.2 代码结构

代码结构。（方盛俊）

### 8.3 运行代码

如何运行代码。（曹明隽）

### 8.4 执行测试

如何执行测试。（曹明隽）


## 九、性能测试

在完成代码后测试，然后完成报告。（曹明隽）

### 9.1 在 test_case0 上的运行情况

包括模型推理准确性以及推理速度等。

### 9.2 在测试数据集上的运行情况

包括模型推理准确性以及推理速度等。

### 9.3 在收集数据集上的运行情况

包括模型推理准确性以及推理速度等。

### 9.4 迁移场景的完成情况

可能会和上一小节合并。


## 十、小组分工

分工情况。（方盛俊）


## 十一、参考文献

参考文献。（曹明隽、任圣杰、方盛俊）

