# 企业智能知识库 RAG 系统

当前版本完成第一阶段：企业文档的加载、清洗、分块、向量化和幂等持久化。问答、引用、拒答和评测将在后续阶段加入。

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
python -m scripts.search_knowledge "物流通常需要几天？" --top-k 3 --min-relevance 0.4
```

结果会返回命中的正文、相关度、文件名、路径、页码和文本块序号。
Chroma 的 cosine distance 越小越相似，本项目用
`relevance_score = 1 - distance` 转成越大越相关的分数，并限制在 `[0, 1]`。

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

## 当前边界

- 扫描版PDF没有文字层时无法直接提取，需要后续OCR管道；
- 当前完成离线入库与检索，还不能向用户生成答案；
- Chroma适合本地项目演示，生产部署需要额外设计备份、并发和访问控制；
- 演示制度均为虚构数据，不代表任何真实企业政策。
