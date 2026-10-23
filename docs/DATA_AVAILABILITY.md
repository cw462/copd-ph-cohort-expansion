# 数据可用性声明

本仓库公开经过范围筛选的分析代码、方法说明、变量定义和汇总结果，用于审阅队列整理与探索性内部评估的方法。公开材料不包含患者级原始数据或可用于重建逐例临床记录的关联信息。

| 材料类别 | 本次公开范围 |
| --- | --- |
| 分析代码与方法 | 公开通用分析逻辑、特征集合、评估设置及方法边界 |
| 汇总结果 | 公开队列计数、汇总性能指标及解释限制 |
| 临床原表与逐例数据 | 不公开，包括身份关联表、逐例标签、临床数值、检查日期和定量特征矩阵 |
| 医学影像与分割 | 不公开原始影像、转换影像、分割掩膜或逐例影像报告 |
| 逐例评估与内部溯源材料 | 不公开逐例预测、折分成员、原表行号、原始文件路径、检查UID或原始数据文件哈希 |

原患者数据含有敏感医学信息，本次发布不包含这些数据。尚未建立可供外部使用的受控数据获取流程，也未指定数据申请联系人、审查机制、审批时限或访问条件。因此，本仓库不作“可按合理请求获取”或保证提供原始数据的承诺。

本次发布未指定软件或数据许可证；不据此声明原患者数据已获得公开或再分发许可，也不声明已经取得未被核实的伦理审批或数据共享授权。

公开代码和汇总结果支持检查方法与汇总口径。由于训练所需的患者级特征、标签和关联证据未公开，仅凭本仓库不能独立重算原队列的交叉验证结果。汇总计数不构成可直接训练的数据集，候选关联也不构成已确认的临床标签。有关分析范围、来源与协议混杂、对照不足及缺少外部验证的限制，见[方法与分析边界](METHODS.md)。

English statement: This repository shares analysis code, methodological descriptions, feature definitions and aggregate results. Patient-level clinical data, images, segmentations, feature matrices, linkage records and individual predictions are not publicly released. A controlled-access process has not been established, and this release makes no commitment to provide patient-level data on request. No software or data license has been assigned to this release. Public files alone are insufficient to reproduce the original patient-level cross-validation results. The analyses are exploratory and have not been externally validated.
