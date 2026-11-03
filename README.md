# 短时降水预报模型与数据处理源码展示

本仓库用于张轩松的实习申请，展示本人结合 GPT 辅助开发的短时降水预报研究中的模型与工程实现。它从原项目中选取模型、数据处理、评分和单元测试源码，以全新历史公开；比赛原始数据、训练权重、平台接口凭据、提交文件、平台截图与历史运行日志均不包含在内。

## 可核实的实现

- PyTorch 多源输入模型，包含观测、雷达、数值天气预报及先验分支。
- 六帧因果 ConvGRU 序列聚合，包含输入形状与有效性检查；有重复最新帧的对照路径。
- 数据缓存读取、归一化、缺测掩码与时间顺序约束。
- 评分计算和隔离的模型单元测试。

## 源码导航

- `src/precip_forecast/models/autonomous_seqgru.py`：六帧 ConvGRU 和继承结构。
- `src/precip_forecast/models/task65_seq4.py`：序列输入与多源融合。
- `src/precip_forecast/realdata/dataset.py`：数据集读取与缓存校验。
- `src/precip_forecast/data/normalization.py`：归一化处理。
- `tests/test_unittest_autonomous_seqgru.py`：模型形状、因果顺序与对照约束。

## 本地查看与测试

推荐 Python 3.10 或以上，在独立虚拟环境中安装依赖。PyTorch 的 CPU/CUDA 安装方式请按其官方说明选择。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python -m unittest discover -s tests -v
```

源码精选包含所选模块的项目内依赖。真实训练和比赛复现仍需有权使用的数据、对应缓存、配置及模型权重；这里的模型测试不能替代真实预报效果验证。

## 历史实验口径

原项目保存的最佳官方客观分为 41.2526，提交时间记录为 2026 年 10 月 4 日。该数值是历史记录，不能用作实时榜单或最终名次。后续数量头实验没有取得可晋级结果；源代码的存在不代表该模型优于所有基线。招聘展示重点是模型实现、实验对照、数据处理和评估方法。

本次公开整理于 2026 年 11 月 3 日运行了所附 ConvGRU 单元测试，20 项全部通过；使用 CPU 和构造输入，不需要真实比赛数据。本次未重新训练或进行官方提交。原始数据与权重的可恢复性不由本展示仓库保证。未另行授予整体开源许可证；第三方框架按其各自许可证使用。
