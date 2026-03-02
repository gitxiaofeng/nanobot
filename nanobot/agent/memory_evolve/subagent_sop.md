# Subagent 调用 SOP

## Task Mode 文件IO协议

- 目录：`temp/{task_name}/`（相对代码根GenericAgent/），主agent cwd在temp/时即 `./{task_name}/`
- 启动：`python agentmain.py --task {task_name} [--llm_no N]`（cwd=代码根）
- 流程：写 input.txt → 启动 → 轮询 output.txt → 读回复 → 写 reply.txt 继续 → 不写则5min自动退出
- input.txt原则：写目标+约束，可指定SOP名。禁写具体实现步骤——除非主agent已读过该SOP确认正确。凭印象猜的步骤会误导subagent
- output.txt：首轮对话的流式输出（持续append），用mtime/size判断更新
- output1.txt, output2.txt...：reply后各轮的流式输出（递增编号），同样持续append

## 后台调用要点

```python
task_dir = os.path.join(agent_root, 'temp', task_name)
proc = subprocess.Popen(
    [sys.executable, 'agentmain.py', '--task', task_name],
    cwd=agent_root, creationflags=0x08000000,
    stdout=open(os.path.join(task_dir, 'stdout.log'), 'w', encoding='utf-8'),
    stderr=open(os.path.join(task_dir, 'stderr.log'), 'w', encoding='utf-8'))
```

- 必须 Popen，禁止 subprocess.run（会阻塞）
- stdout.log/stderr.log 用于调试subagent卡死、LLM调用失败等问题
- `--llm_no` 默认=sonnet 4.5，`--llm_no 1`=opus 4.6
- 文件统一 UTF-8，subagent 无 reply 5min 自动退出无需清理

## 场景1：测试模式 - 行为验证
**用途**：观察agent真实行为，修正RULES/L2/L3/SOP
**流程**：创建test_path/写input.txt→启动subagent→轮询output.txt(2秒间隔)→验证→清理重复
**测试原则**：只给目标，不提示位置/不诱导做法，观察自主选择
**修正闭环**：发现问题→设计测试→定位根源(RULES/L2/L3/SOP)→patch修正→验证
**技术要点**：Insight优先级>SOP；subagent的cwd=temp/
**两种测试**：
- 测SOP质量：input指定SOP名（如"用ezgmail_sop查看最近3封未读邮件"），排除导航干扰，失败即SOP问题
- 测导航能力：input只写目标，验证subagent能自主从insight找到正确SOP。禁止内联SOP内容

## 场景2：Map模式 - 并行处理
**用途**：将N个独立同构子任务分发给各自的subagent处理
**核心优势**：独立上下文。避免处理文档A的长上下文污染处理文档B的质量
**约束**：
- 文件系统共享是优点：不同agent处理不同输入文件，产生不同输出文件
- 共享资源冲突：键鼠/浏览器主体不可共享（浏览器可分tab但需谨慎），subagent任务应限于文件处理
- 不满足map模式的任务 → 主agent顺序执行即可，别用subagent
**标准流程（map-reduce）**：
1. 主agent准备阶段：爬取/dump数据，存为多个独立输入文件
2. 分发：对每个文件启动一个subagent处理（主agent自己也可以处理其中一个）
3. 收集：等所有subagent完成，主agent读取各输出文件，汇总结果