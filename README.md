# Agentic-RAG：图谱增强检索临床辅助决策系统

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-5.18-green)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1EA)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-red)

## 项目简介

Agentic-RAG 是面向临床场景的**图谱增强检索生成（GraphRAG）**系统。融合**向量语义检索**、**知识图谱推理**和**对话记忆**三种能力，对历史临床问答数据进行智能检索与答案生成，为医生提供辅助决策支持。

核心思路：将历史 QA 数据同时构建为向量索引和 Neo4j 知识图谱，通过 LLM 驱动的智能路由动态选择最佳检索策略，结合对话记忆追踪诊断推理链，最终生成可追溯的医学回答。

## 系统架构

```
                    ┌──────────────────────────┐
                    │   用户界面 (CLI / Streamlit)   │
                    └─────────────┬────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │   ConversationMemory      │  ← 对话记忆（缓冲 + 图谱 + 向量）
                    └─────────────┬────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │  IntelligentQueryRouter   │  ← LLM 分析复杂度 → 选择策略
                    └──────┬────────┬──────────┘
                           │        │
              ┌────────────┘        └────────────┐
              ▼                                  ▼
   ┌──────────────────┐              ┌──────────────────┐
   │ HybridRetrieval   │              │ GraphRAGRetrieval │
   │ 向量 + BM25 + 图   │              │ 多跳 / 子图 / 路径  │
   └───────┬──────────┘              └───────┬──────────┘
           │                                  │
     ┌─────┴─────┐                      ┌─────┴─────┐
     ▼           ▼                      ▼           ▼
   Milvus      Neo4j                  Neo4j       Neo4j
  (向量库)    (图数据库)               (图数据库)    (图数据库)
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
            ┌──────────────────────────┐
            │ GenerationIntegration     │  ← DeepSeek 生成 + 医疗免责声明
            │ (记忆上下文 + 检索结果)     │
            └──────────────────────────┘
```

## 功能特性

### 检索与推理
- **三通道检索**：向量语义（Milvus + BGE） + 关键词（BM25） + 图谱推理（Neo4j Cypher）
- **智能路由**：LLM 分析查询复杂度与关系密度，自动选择混合检索 / 图谱推理 / 融合策略
- **多跳推理**：支持疾病 → 症状 → 药物 → 禁忌的深度图遍历

### 对话记忆（NEW）
- **三层混合记忆**：短期缓冲（deque） + 图谱记忆（Neo4j Session/Turn） + 语义记忆（Milvus）
- **诊断链追踪**：会话中讨论的疾病、策略、推理路径持久化为图结构，可审计可回溯
- **跨会话召回**：语义相似病例跨会话检索，回答"之前见过类似病例吗？"
- **指代消解**：理解"刚才提到的那个药"、"之前的诊断"等上下文指代

### 数据与工程
- **增量更新**：支持运行时导入新 CSV，追加到图谱和向量索引
- **智能同步检测**：启动时对比 Neo4j + Milvus 元数据，跳过已同步数据的重复构建
- **编译正则疾病匹配**：1000+ 疾病词典 O(N) 单次扫描，替代 O(N×D) 暴力循环
- **索引构建轮询**：替代硬编码 sleep，轮询 Milvus 索引进度直到完成
- **流式输出**：DeepSeek API 流式生成，实时展示推理过程
- **双界面**：Streamlit Web UI + Python CLI

## 技术栈

| 类别       | 技术                                                     |
| ---------- | -------------------------------------------------------- |
| 大语言模型 | DeepSeek Chat（OpenAI 兼容 API）                         |
| 向量数据库 | Milvus 2.5（HNSW 索引，Cosine 相似度）                   |
| 图数据库   | Neo4j 5.18（APOC 插件）                                   |
| 嵌入模型   | BAAI/bge-base-zh-v1.5（768 维）                          |
| 框架       | LangChain, HuggingFace Transformers, Sentence-Transformers |
| 前端       | Streamlit                                                |
| 基础设施   | Docker Compose（etcd + MinIO + Milvus + Neo4j）          |

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- CUDA（可选，GPU 加速嵌入）

### 1. 克隆与配置

```bash
git clone <repo-url>
cd Agentic-RAG
```

编辑 `.env`，填入 DeepSeek API key：
```env
DEEPSEEK_API_KEY=sk-your-key-here
```

### 2. 启动服务

```bash
docker compose up -d
```

服务端口：Milvus `19530`，Neo4j Bolt `8687`，Neo4j HTTP `8474`

### 3. 安装依赖

```bash
conda activate all-in-rag   # 或创建新环境
pip install -r requirements.txt
```

### 4. 下载嵌入模型（可选，已内置）

```bash
python scripts/download_model.py
```

### 5. 运行

**CLI 交互模式：**
```bash
python app/main.py
```

交互命令：
| 命令 | 说明 |
|------|------|
| `<直接输入问题>` | 发送临床咨询 |
| `stats` | 查看知识库与记忆模块统计 |
| `rebuild` | 重建知识图谱与向量库 |
| `add <路径>` | 增量导入 CSV 数据 |
| `quit` | 安全退出（自动持久化记忆） |

**Web UI 模式：**
```bash
streamlit run app/web_app.py
```

## 项目结构

```
Agentic-RAG/
├── .env                  # API key 等环境变量
├── .env.example          # 环境变量模板
├── requirements.txt      # Python 依赖
├── docker-compose.yml    # 全部服务编排
│
├── app/                  # 主应用
│   ├── main.py           # 系统入口 & 协调器
│   ├── web_app.py        # Streamlit Web UI
│   ├── config.py         # 全局配置 (GraphRAGConfig)
│   ├── clean_milvus.py   # 清理 Milvus 集合
│   └── rag_modules/
│       ├── graph_data_preparation.py      # CSV 导入 & Neo4j 图谱构建
│       ├── milvus_index_construction.py   # Milvus 向量索引构建
│       ├── graph_indexing.py              # 内存图索引 (KV 存储)
│       ├── hybrid_retrieval.py            # 混合检索 (向量+BM25+图)
│       ├── graph_rag_retrieval.py         # 图谱检索 (多跳/子图/路径)
│       ├── intelligent_query_router.py    # LLM 查询路由引擎
│       ├── generation_integration.py      # DeepSeek 答案生成
│       └── conversation_memory.py         # 对话记忆模块 (三层混合)
│
├── models/               # 本地嵌入模型
│   ├── bge-base-zh-v1.5/
│   └── bge-m3/
│
├── data/
│   ├── raw/              # 原始 CSV 数据 (6 科室 ~86k QA)
│   ├── processed/         # 处理后数据
│   ├── disease_dict.txt   # 疾病词典 (~1000 条目)
│   ├── test_100.csv        # 测试数据
│   ├── images/            # 医学流程图
│   └── neo4j/             # Neo4j 持久化 (gitignored)
│
├── scripts/              # 工具脚本
│   ├── download_model.py
│   ├── convert_csv_encoding.py
│   └── clean_neo4j.py
│
└── volumes/              # Docker 运行时数据 (gitignored)
```

## 模块详解

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据准备 | `graph_data_preparation.py` | CSV 递归加载，编译正则在 O(N) 时间内匹配疾病实体，批量构建 Neo4j 图谱（Department→Consultation→Disease），合并 doc+chunk 单次遍历生成向量文本 |
| 向量索引 | `milvus_index_construction.py` | BGE 模型嵌入编码，Milvus Collection 管理，HNSW 索引，分批防 OOM 插入，轮询等待索引就绪 |
| 图索引 | `graph_indexing.py` | 内存 KV 存储实体/关系，关键词查找，邻居扩展，去重 |
| 混合检索 | `hybrid_retrieval.py` | 双层检索（实体级+主题级），LLM 提取关键词，BM25 + Milvus 向量 + Neo4j 补充，Round-robin 融合 |
| 图谱检索 | `graph_rag_retrieval.py` | 查询意图理解（entity_relation/multi_hop/subgraph/path_finding/clustering），Cypher 多跳遍历，子图提取，图结构推理链 |
| 查询路由 | `intelligent_query_router.py` | LLM 分析复杂度+关系密度，路由到 hybrid/graph/combined 策略，失败降级规则匹配 |
| 答案生成 | `generation_integration.py` | 记忆上下文+检索结果融合 prompt，流式/非流式生成，网络重试，强制医疗免责声明 |
| **对话记忆** | `conversation_memory.py` | 短期缓冲（deque）+ 图谱记忆（Neo4j Session/Turn 节点）+ 语义记忆（Milvus 新集合），三层检索融合 |

## 记忆模块数据模型

```
(Session {session_id, start_time, turn_count, patient_summary})
    └─[:CONTAINS]→ (Turn {turn_id, question, answer, strategy, complexity_score})
        ├─[:REFERENCES_DISEASE]→ (Disease)   ← 复用已有疾病节点
        ├─[:REFERENCES_DEPT]→ (Department)   ← 复用已有科室节点
        └─[:NEXT]→ (Turn)                    ← 链式推理时序
```

## 配置说明

全部配置在 `app/config.py` 的 `GraphRAGConfig` 数据类中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `neo4j_uri` | `bolt://localhost:8687` | Neo4j 连接 |
| `milvus_host` / `milvus_port` | `localhost:19530` | Milvus 连接 |
| `embedding_model` | `<auto>/models/bge-base-zh-v1.5` | 嵌入模型路径 |
| `llm_model` | `deepseek-v4-flash` | 生成模型 |
| `top_k` | `5` | 检索返回数 |
| `temperature` | `0.1` | 生成温度 |
| `memory_enabled` | `True` | 启用记忆模块 |
| `memory_buffer_size` | `10` | 短期缓冲轮数 |
| `memory_top_k` | `3` | 语义记忆召回数 |
| `memory_summary_threshold` | `10` | 触发 LLM 摘要轮数 |

## 数据处理流程

```
CSV 加载 (data/processed/)
  │
  ├─→ 疾病识别 (编译正则单次扫描, O(N))
  │     └─→ Neo4j 图谱构建 (UNWIND 批量写入, batch=2000)
  │           └─→ SystemMeta 元数据标记
  │
  └─→ 文档块构建 (doc+chunk 合并单次遍历)
        └─→ Milvus 向量索引 (batch=500, 轮询索引就绪)
              └─→ 检索引擎初始化 (内存图索引预热)
                    └─→ 记忆模块初始化 (Neo4j 约束 + Milvus 新集合)
```

## License

MIT
