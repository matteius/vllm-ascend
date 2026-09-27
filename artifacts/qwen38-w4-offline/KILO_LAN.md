# W4 Kilo 启动（2026-09-27）

## 本轮范围

按用户要求使用既有端口8001，不使用8002，不改路由、防火墙或增加代理。
保持private W4 runtime、算子、MTP k2、TP4+EP、
FULL_DECODE_ONLY [1,2,3,6]、chunk512、max-seqs2、
max-model-len262144、utilization .94 / KV fraction .80，不升级依赖。
host=0.0.0.0，开启auto-tool-choice / qwen3_xml。
W8 home launcher、runtime、权重文件不变；当前8001运行W4，不是W8。

## 地址和边界

- base URL：`http://192.168.53.187:8001/v1`
- model ID：`qwen38-w4-experimental`
- 此前给出的Kilo W4 entry应使用上述8001地址；本轮未修改Kilo配置文件。
- 配置上限：两个并发session，每个262144 tokens，包含输入和输出。
- 完整2×262144 soak未完成：r6第一条prefill约113664tokens时触发
  swap增长guard并取消client，不是NPU OOM。详见`BATCHING_DUAL262.md`。
- 未新增API鉴权，仅供可信局域网使用，不要转发到公网。
- 长cold prefill仍慢；此前Kilo entry的超时为7200000ms。

## 运行记录

host：`matteius@192.168.53.187`。
taskdir：`/srv/ai/src/qwen38-w4-hardware-20260927`。
tmux：`qwen-w4-kilo-r8`，API468614，engine469358。
workers469731–469734，PID-tree验证及affinity已完成。
旧8002实例API435927已TERM，确认旧进程树退出、NPU无运行进程后启动8001。
r7的3个text smoke及双session短测通过，但客户端到8002连接超时；
按用户指示立即停掉r7并改为8001，不继续排查或修改网络。

启动命令（本轮已运行，不要同时重复启动）：

```bash
bash /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-kilo-server-r8.sh
```

wrapper保留thermal watchdog及server进程，日志路径有不覆盖检查。
再次运行应使用新run-id日志路径。启动脚本存档在`capacity-r1/`。
smoke结束后使用`qwen38-w4-kilo-affinity-r8.py`校验PID-tree，绑定worker线程到
0–5 / 8–13 / 16–21 / 24–29。此脚本只接受W4模型、test venv、8001实例。

```bash
curl -fsS http://192.168.53.187:8001/v1/models
python3 artifacts/qwen38-w4-offline/capacity-r1/qwen38-w4-kilo-tools-r7.py
```

工具验证只返回合成access-code，不执行模型请求的任何系统操作。
不重复full-context soak，不把短窗口smoke视为dual262完整验证。

## r8 验证状态

real-weight启动通过。来自Kilo主机192.168.50.211的8001 HTTP请求成功；
8002已无监听。三个text smoke（乘法、1–50、Python输出）均正确完结。
streaming auto-tool单session（thinking=false）及两个并行session（thinking=true）
共三条tool-call/result round trip均通过，独立access-code正确。
FULL3及FULL6运行计数确认replay；draft290 / accepted280，零preemption，
工具检查结束时running=0 / waiting=0，服务保留给用户。
短请求acceptance不能代表coding长上下文acceptance或吞吐。

证据：`capacity-r1/http-smoke-kilo-r8.jsonl`、`kilo-r8-tool-smoke.jsonl`、
`kilo-r8-affinity.json`、`server-kilo-r8-gates.log.gz`。
r7相同配置（仅端口不同）的single256+512为16.107tok/s，
dual256+256为10.147/9.612tok/s，decode overlap24.790s，零preemption。
本轮r8不再重复吞吐或full-context soak，避免占用用户测试时间。

本轮不运行dummy、全库UT或准确率评测；未修改推理源代码。
language-model-only；multimodal及flashcomm1不在此次启动范围。
保留max-seqs2的小batch kernel限制，不宣称128k+bs16支持。

## 第三个完整窗口的只读分析

当前shared KV规划容量1,215,660tokens；3×262144=786432tokens，
因此KV分配数学上有余量，不需要复制模型权重或立即提高utilization。
每rank npu-smi约39,830–40,802MiB，容量46,717–47,302MiB；
torch allocated36.60GiB / reserved37.07GiB。部分已分配KV尚未被请求使用，
不能把npu-smi空闲直接当成新增session容量。

实际障碍是MTP k2的三个session需要9个验证token、top-k10共90routes。
当前W4 routed路径上限80routes，超过后非capture路径转host routing，
capture直接报错；现有graphs只capture[1,2,3,6]。需先扩展或分批处理
九token的device-routed路径，再验证九token图、三session状态隔离、
host snapshots及长窗口吞吐。不能只把max-num-seqs改3就宣称支持。
当前仍保持两session；完整dual262尚未验证，更不能提前声称triple262通过。
