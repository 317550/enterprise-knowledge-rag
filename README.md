# 企业智能知识库 RAG 系统

当前版本完成文档入库、混合检索、交叉编码器精排和带来源引用的RAG答案生成。系统在没有可靠知识片段或模型引用无效时拒绝回答，避免把模型猜测伪装成企业制度。

## 完整架构

```text
PDF / Markdown / TXT
        ↓
受控加载与PDF逐页解析
        ↓
保守文本清洗
        ↓
带重叠的边界感知分块
        ↓
BGE文档向量
        ↓
Chroma持久化与版本同步
        ↓
用户问题 ──┬── BGE稠密召回 ──┐
           └── BM25词法召回 ─┤
                              ↓
                         RRF排名融合
                              ↓
                     CrossEncoder精排
                              ↓
                     Top-1查询级门控
                              ↓
              去重 / 重叠裁剪 / 字符预算
                              ↓
                  DeepSeek受约束答案生成
                              ↓
                     引用校验 / 可靠拒答
```

工程特性：

- PDF保留原始页码，为后续引用溯源提供依据；
- 文件路径生成稳定`document_id`；
- 文件内容生成`checksum`；
- 文本块内容及位置生成稳定`chunk_id`；
- 文件没有变化时跳过向量计算；
- 文件更新后写入新块并清理旧块；
- 单个损坏文件不会中断整批入库；
- 测试使用假向量模型，不下载BGE模型。

## 工程结构

```text
app/
├── loaders.py              # 受控文档加载
├── chunking.py             # 边界感知分块
├── embeddings.py           # BGE向量适配器
├── vector_store.py         # Chroma持久化与稠密召回
├── lexical_retriever.py    # jieba与BM25词法召回
├── hybrid_retriever.py     # RRF融合
├── reranker.py             # CrossEncoder精排
├── context_builder.py      # 上下文去重与预算
├── rag_service.py          # 检索、生成和引用校验
└── evaluation_service.py   # 离线指标计算
scripts/
├── ingest_documents.py
├── search_knowledge.py
├── ask_knowledge.py
└── evaluate_rag.py
evaluation/
└── questions.jsonl         # 39条固定评测用例
```

## 安装

```cmd
conda activate enterprise_rag
python -m pip install -r requirements.txt
copy .env.example .env
```

首次执行真实入库时，Sentence Transformers会下载`BAAI/bge-small-zh-v1.5`。模型文件可能较大，请保证网络和磁盘空间充足。

## 执行入库

将企业文档放入`data/raw`，然后执行：

```cmd
python -m scripts.ingest_documents
```

首次运行示例：

```json
{
  "files_seen": 4,
  "files_inserted": 4,
  "files_replaced": 0,
  "files_skipped": 0,
  "chunks_written": 4,
  "errors": []
}
```

再次运行时，未变化文件会被跳过。实际文本块数量取决于文档长度和分块配置，不应把示例数字写成固定测试条件。

## 执行检索

完成至少一次文档入库后执行：

```cmd
python -m scripts.search_knowledge "商品签收后多久可以退货？"
```

也可以临时覆盖召回数量和最低相关度：

```cmd
python -m scripts.search_knowledge "物流通常需要几天？" --top-k 3 --min-relevance 0.05
```

检索管道会先分别执行BGE稠密召回和BM25关键词召回，使用RRF融合两路排名，再由CrossEncoder对候选进行精排。结果会返回正文、来源元数据以及`vector_score`、`bm25_score`、`rrf_score`和`rerank_score`。这些字段是不同阶段的检索分数，不是概率，也不能直接横向比较。

第一次执行检索时会额外下载`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。可以在`.env`中配置：

```env
HYBRID_CANDIDATE_K=20
RRF_K=60
RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANK_BATCH_SIZE=16
```

## 执行问答

在本地`.env`中填写DeepSeek密钥：

```env
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=800
RAG_MAX_CONTEXT_CHARS=6000
RAG_MAX_CHUNKS_PER_SOURCE=2
```

不要提交`.env`，仓库只保留不含真实密钥的`.env.example`。

完成文档入库后执行：

```cmd
python -m scripts.ask_knowledge "商品签收后多久可以退货？"
```

也可以覆盖检索参数：

```cmd
python -m scripts.ask_knowledge "物流通常需要几天？" --top-k 3 --min-relevance 0.5
```

问答管道：

```text
用户问题 → BGE与BM25召回 → RRF融合 → CrossEncoder精排
        → Top-1查询级门控 → 上下文编号 → DeepSeek结构化生成
        → 引用编号校验 → 答案与来源
```

相关度阈值用于判断整个问题是否存在可靠候选：Top-1低于阈值时返回空结果；Top-1通过门槛后保留完整Top-K，避免多来源问题中的必要补充资料因单块分数较低而被误删。当前`0.05`来自固定评测集上的自动阈值扫描，不应直接照搬到其他模型或业务数据。

模型只负责选择检索上下文并生成答案。独立的上下文构建器会按`chunk_id`和正文去重、移除同一来源相邻块的重复前后缀、限制单个来源的块数量，并严格控制字符预算。程序会独立校验引用编号；没有命中、模型主动判断资料不足、引用越界或答案缺少引用标记时，统一返回拒答结果。

## 测试

```cmd
python -m pytest -v
```

当前版本共57项自动化测试。

测试覆盖：

- 文本规范化与段落保留；
- UTF-8 Markdown加载；
- PDF页码保留；
- 非法文件类型与越界路径拦截；
- 分块长度和ID稳定性；
- 首次入库；
- 重复入库跳过嵌入；
- 文档变更后清理旧文本块。
- Top-K检索顺序、来源元数据和相关度阈值；
- 空知识库、空问题和非法检索参数。
- BM25词法召回、RRF融合去重和CrossEncoder精排；
- 带引用答案及来源映射；
- 无命中时不调用大模型；
- 模型判定资料不足时拒答；
- 缺失、越界或未落入正文的引用被拒绝；
- 上下文字符预算和提示注入隔离。
- 上下文去重、重叠裁剪和单来源数量限制；
- Hit@1、Hit@3、MRR、多来源召回和拒答指标计算；

## 离线评测数据集

`evaluation/questions.jsonl`保存固定评测问题，不参与向量入库。当前数据集包含可回答问题和不可回答问题，并覆盖：

- 语义改写查询；
- 精确数字、术语和制度名称查询；
- 需要多个制度共同支持的查询；
- 知识库没有依据、应当拒答的查询。

每条记录声明预期来源与答案关键词。`app/evaluation_dataset.py`负责加载并验证数据契约，`app/evaluation_service.py`计算指标。

仅运行检索评测，不调用DeepSeek：

```cmd
python -m scripts.evaluate_rag --mode retrieval
```

运行完整问答评测，会对39条用例调用DeepSeek：

```cmd
python -m scripts.evaluate_rag --mode rag
```

也可以一次执行两类评测：

```cmd
python -m scripts.evaluate_rag --mode all
```

完整报告分别保存到：

```text
evaluation/results/retrieval_report.json
evaluation/results/rag_report.json
```

检索报告给出纯向量、BM25与向量的RRF融合、加入CrossEncoder精排后的三阶段消融结果，并保存逐候选的向量、BM25、RRF和Reranker分数。报告包含Hit@1、Hit@3、MRR、`source_recall_at_3`、不可回答问题空结果率和自动阈值扫描结果。RAG报告包含可回答成功率、拒答准确率、引用精确率、引用召回率、答案关键词覆盖率以及逐题答案和关键词审计字段。

其中Hit@K表示前K个文本块中至少出现一个预期来源；`source_recall_at_3`用于衡量多来源问题在前三个文本块中的来源覆盖。RAG指标采用确定性规则计算：拒答标志、引用来源和预先声明的答案关键词，不使用另一个大模型充当裁判。

## 固定评测结果

以下结果来自39条固定回归用例，其中33条可回答、6条不可回答。排名评测关闭业务阈值，拒答阈值通过查询级门控扫描得到。

| 检索阶段 | Hit@1 | Hit@3 | MRR | 来源召回@3 |
|---|---:|---:|---:|---:|
| BGE纯向量 | 90.91% | 96.97% | 92.93% | 95.45% |
| BGE + BM25 + RRF | 96.97% | 100% | 97.98% | 100% |
| RRF + CrossEncoder | 96.97% | 100% | 98.48% | 100% |

阈值扫描在`0.03`至`0.07`区间同时保持可回答问题Hit@3、来源召回@3和不可回答问题空结果率为100%，系统选择平台区间中点`0.05`。

| 端到端RAG指标 | 结果 |
|---|---:|
| 可回答问题成功率 | 100%（33/33） |
| 不可回答问题拒答准确率 | 100%（6/6） |
| 引用精确率 | 100% |
| 引用召回率 | 100% |
| 答案关键词覆盖率 | 100% |
| 执行失败 | 0 |

这些指标用于当前演示语料和固定用例的工程回归，不代表未知生产流量上的泛化准确率。评测集同时参与了阈值选择，因此不能将结果表述为独立测试集成绩。

## 核心技术选型

```text
Embedding: BAAI/bge-small-zh-v1.5
Lexical retrieval: jieba + BM25
Fusion: Reciprocal Rank Fusion
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Generation: DeepSeek
Vector store: Chroma
```

## 当前边界

- 扫描版PDF没有文字层时无法直接提取，需要后续OCR管道；
- 当前为命令行问答，尚未提供FastAPI或Web界面；
- 引用校验能限制来源编号，但不能完全消除大模型对资料内容的错误概括；
- Chroma适合本地项目演示，生产部署需要额外设计备份、并发和访问控制；
- 演示制度均为虚构数据，不代表任何真实企业政策。
- 当前39条评测用例适合项目回归和模型对比，不代表生产环境的统计显著性；
- Reranker阈值需要针对具体模型和真实业务查询重新校准；
- 命令行每次运行都会重新加载本地模型，生产服务应常驻进程并在启动阶段预热。
