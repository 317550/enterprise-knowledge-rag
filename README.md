# 企业智能知识库 RAG 系统

当前版本完成文档入库、混合检索、交叉编码器精排和带来源引用的RAG答案生成。系统在没有可靠知识片段或模型引用无效时拒绝回答，避免把模型猜测伪装成企业制度。

## 第一阶段架构

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

## 安装

```cmd
conda activate enterprise_rag
python -m pip install -r requirements.txt
copy .env.example .env
```

首次执行真实入库时，Sentence Transformers会下载`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。模型文件可能较大，请保证网络和磁盘空间充足。

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
python -m scripts.search_knowledge "物流通常需要几天？" --top-k 3 --min-relevance 0.4
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
        → 阈值过滤 → 上下文编号 → DeepSeek结构化生成
        → 引用编号校验 → 答案与来源
```

模型只负责选择检索上下文并生成答案。程序会独立校验引用编号；没有命中、模型主动判断资料不足、引用越界或答案缺少引用标记时，统一返回拒答结果。

## 测试

```cmd
python -m pytest -v
```

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

## 离线评测数据集

`evaluation/questions.jsonl`保存固定评测问题，不参与向量入库。当前数据集包含可回答问题和不可回答问题，并覆盖：

- 语义改写查询；
- 精确数字、术语和制度名称查询；
- 需要多个制度共同支持的查询；
- 知识库没有依据、应当拒答的查询。

每条记录声明预期来源与答案关键词。`app/evaluation_dataset.py`负责加载并验证数据契约，`tests/test_evaluation_dataset.py`检查问题ID、问题文本、来源文件、类型覆盖和不可回答样例。Hit@1、Hit@3、MRR及拒答准确率将在最终评测阶段由统一脚本计算。

## 当前边界

- 扫描版PDF没有文字层时无法直接提取，需要后续OCR管道；
- 当前为命令行问答，尚未提供FastAPI或Web界面；
- 引用校验能限制来源编号，但不能完全消除大模型对资料内容的错误概括；
- Chroma适合本地项目演示，生产部署需要额外设计备份、并发和访问控制；
- 演示制度均为虚构数据，不代表任何真实企业政策。
