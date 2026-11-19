# Harmony Risk Guard v0.2 合成数据集

共 288 条合成样本，覆盖文件操作、权限修改、敏感数据外发三类，每类 12 个基础场景 × 8 个文本变体。按 group_id 分组切分：train 144 / val 72 / test 72。

文件：train.jsonl、val.jsonl、test.jsonl、label_policy.md。

## 字段
- sample_id: 唯一 ID
- group_id: 基础场景 ID（切分防泄漏）
- scenario: 场景类别
- user_instruction: 用户指令或 Agent 操作描述
- agent_tool: 抽象工具名称
- tool_arguments: 操作目标、范围、数量、目的地
- context: 授权、数据分级、可恢复性、备份、接收方可信度
- label: CONTINUE / ASK / BLOCK
- reason: 标注原因（训练时不要放进输入，避免答案泄漏）
- source_type: synthetic
- requires_human_review: true

## 训练输入建议
将 user_instruction、agent_tool、tool_arguments、context 规范化序列化为文本；仅用 label 作监督标签。评测时禁止使用 reason 作为模型输入。

## 注意
这些是人工编写规则驱动的合成案例，未进行人工安全专家审核；它们用于工程穿刺，而不是可靠的生产安全判别。
