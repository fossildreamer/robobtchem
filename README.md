# 电池极片正反面识别

本项目用于识别料盘或散料区域图像中的圆形电池极片，并判断每个极片的正反面。项目参考作业页面中的“锂电池极片自动摆盘与翻面系统”场景，但本项目只实现视觉识别部分，不包含抓取、翻面、摆盘、PLC、机械臂或三轴平台控制。

当前样例图中，银白/金属反光面按正面处理，黑色面按反面处理。

## 已实现功能

- 支持单张图片、图片文件夹和摄像头输入
- 自动检测圆形极片位置
- 根据颜色和亮度特征判断正面、反面或未识别
- 在原图上绘制识别圆、中心点、类别和置信度
- 输出 `results.csv` 结果表格
- 支持中文路径图片读写
- 使用独立 Python 环境 `disk_reg`，不改动主 Python 环境

## 项目结构

```text
电池极片正反面识别/
├── disk_reg/             # 独立 Python 虚拟环境，不提交
├── data/
│   ├── frames/           # 当前测试图片
│   ├── raw/              # 可放原始图片
│   └── samples/          # 可放示例图片
├── outputs/
│   ├── images/           # 标注后的图片
│   ├── debug/            # 预留调试输出
│   └── results.csv       # 识别结果表格
├── src/
│   ├── main.py           # 命令行入口
│   ├── detector.py       # 极片检测
│   ├── classifier.py     # 正反面分类
│   ├── models.py         # 数据结构
│   └── utils.py          # 图片读写、标注、CSV 输出
├── requirements.txt
└── README.md
```

## 环境安装

项目环境名称为 `disk_reg`。如果环境已经存在，可以直接跳到安装依赖。

```powershell
python -m venv disk_reg
.\disk_reg\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，也可以不激活环境，直接使用环境内的 Python：

```powershell
.\disk_reg\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行示例

识别当前样例图片：

```powershell
.\disk_reg\Scripts\python.exe -m src.main --input data\frames\1.jpg --output outputs
```

识别整个图片文件夹：

```powershell
.\disk_reg\Scripts\python.exe -m src.main --input data\frames --output outputs
```

使用摄像头识别一帧：

```powershell
.\disk_reg\Scripts\python.exe -m src.main --input 0 --mode camera --output outputs
```

也可以直接运行脚本：

```powershell
.\disk_reg\Scripts\python.exe src\main.py --input data\frames\1.jpg --output outputs
```

## 输出结果

运行后会生成：

- `outputs/images/xxx_annotated.jpg`：带识别标注的图片
- `outputs/results.csv`：每个极片的识别结果

CSV 字段包括：

- `image`：图片文件名
- `index`：极片编号
- `center_x`、`center_y`：圆心坐标
- `radius`：半径
- `label`：英文类别，`front` / `back` / `unknown`
- `label_cn`：中文类别，正面 / 反面 / 未识别
- `confidence`：置信度
- `mean_b`、`mean_g`、`mean_r`、`mean_h`、`mean_s`、`mean_v`：颜色统计特征
- `dark_ratio`、`bright_ratio`、`metallic_ratio`、`copper_ratio`：分类辅助特征

## 样例验证

使用 `data/frames/1.jpg` 测试，当前结果为：

```text
detected=16
front=8
back=8
unknown=0
```

标注图保存在：

```text
outputs/images/1_annotated.jpg
```

## 算法说明

程序采用传统视觉方法，不依赖深度学习模型：

1. 使用 Hough 圆检测寻找圆形候选区域。
2. 使用暗色掩膜检测黑色反面。
3. 使用 HSV + Lab 颜色空间检测银白/金属正面，减少纸板背景误判。
4. 合并重复候选圆，优先保留完整连通区域候选。
5. 对每个候选区域提取颜色、亮度、暗色比例、金属比例等特征。
6. 输出正面、反面或未识别结果。

## 参数调整

如果图片分辨率、极片大小或拍摄距离变化较大，可以调整半径比例：

```powershell
.\disk_reg\Scripts\python.exe -m src.main --input data\frames --output outputs --min-radius-ratio 0.02 --max-radius-ratio 0.10
```

常见调整方向：

- 漏检小极片：适当减小 `--min-radius-ratio`
- 把大背景圆误检为极片：适当减小 `--max-radius-ratio`
- 圆片大小变化很大：适当放宽两个半径参数

## 数据采集建议

- 光照尽量均匀，避免强反光直射
- 相机位置固定，减少透视变化
- 背景颜色尽量不要接近极片颜色
- 图片中尽量不要出现其他圆形物体
- 正面、反面样本都要采集，便于后续调阈值

## 参考

- 作业背景页面：[锂电池极片自动摆盘与翻面系统](https://www.robotchem.com/zh/stu26/prj11_disk_place)
