# Agentic-RAG：基于图谱增强检索的临床辅助问答系统

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1EA)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--Chat-red)

## 项目简介

**Agentic-RAG** 是一个面向医疗场景的 **图谱增强检索生成（GraphRAG）** 系统。它融合了**向量检索**与**知识图谱推理**两种范式，能够对临床问答数据进行智能检索与答案生成，为医生和患者提供辅助决策支持。

核心思路：将历史临床 QA 数据同时构建为向量索引和 Neo4j 知识图谱，通过智能路由引擎动态选择最佳检索策略（混合检索、图谱推理或二者融合），最终由大语言模型生成专业、可追溯的医学回答。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面 (Streamlit / CLI)              │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│               IntelligentQueryRouter                     │
│          (LLM 驱动查询路由 → 策略选择)                    │
└──────┬────────────────────────────┬─────────────────────┘
       ▼                            ▼
┌──────────────┐          ┌──────────────────┐
│ HybridRetrieval│         │  GraphRAGRetrieval│
│ (向量 + BM25)  │         │  (多跳 / 子图 /    │
│               │         │   路径推理)         │
└──────┬───────┘          └──────┬───────────┘
       │                         │
       ▼                         ▼
┌──────────────┐          ┌──────────────────┐
│   Milvus     │          │      Neo4j        │
│ (向量数据库)   │          │  (知识图谱)        │
└──────────────┘          └──────────────────┘
       ▲                         ▲
       └──────────┬──────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│               GenerationIntegrationModule                │
│          (DeepSeek 答案生成 + 医疗免责声明)               │
└─────────────────────────────────────────────────────────┘
```

### 数据处理流程

1. **CSV 导入** → 读取各科室历史 QA 数据
2. **疾病实体识别** → 基于疾病词典匹配提取疾病关键词
3. **Neo4j 图谱构建** → 创建 `科室 → 咨询 → 疾病` 关系网络
4. **向量索引构建** → 使用 BGE 模型编码文档，写入 Milvus
5. **检索 & 生成** → 路由 → 检索 → 融合 → DeepSeek 生成答案

## 功能特性

- **双通道检索**：结合向量语义检索（Milvus + BGE）与传统关键词检索（BM25）
- **图谱推理**：支持多跳实体关系查询、子图提取、最短路径发现
- **智能路由**：LLM 动态分析查询复杂度，自动选择混合检索 / 图谱检索 / 融合策略
- **增量扩展**：支持持续添加新问答数据，实时更新图谱与向量索引
- **医疗专属提示词**：内置患者隐私隔离、可追溯引用、专业声明机制
- **流式输出**：支持 DeepSeek API 流式生成，实时展示思考过程
- **交互界面**：提供 Streamlit Web UI 和 Python CLI 两种交互方式

## 技术栈

| 类别        | 技术选型                                                     |
| ----------- | ------------------------------------------------------------ |
| 大语言模型  | DeepSeek Chat（OpenAI 兼容接口）                             |
| 向量数据库  | Milvus 2.5（HNSW 索引，Cosine 相似度）                      |
| 图数据库    | Neo4j 5.x                                                    |
| 嵌入模型    | BAAI/bge-base-zh-v1.5（768维）、BAAI/bge-m3                  |
| 框架工具    | LangChain（BM25Retriever）、HuggingFace Transformers         |
| 前端界面    | Streamlit                                                     |
| 基础设施    | Docker Compose（Milvus 全家桶：etcd + MinIO + standalone）    |

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose（用于 Milvus 和 Neo4j）
- CUDA（可选，用于 GPU 加速嵌入）

### 1. 克隆项目

```bash
git clone <repo-url>
cd Agentic-RAG
```

### 2. 配置环境变量

创建 `config.env` 文件（或重命名已有模板）：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 3. 启动基础设施

```bash
# 启动 Milvus（向量数据库）
docker compose -f code/docker-compose.yml up -d

# 启动 Neo4j（图数据库）
docker compose -f data/C10/docker-compose.yml up -d
```

### 4. 安装依赖

```bash
pip install -r code/requirements.txt
```

### 5. 下载嵌入模型

```bash
python code/m3_download.py
```

### 6. 运行系统

**CLI 交互模式：**
```bash
python code/C10/main.py
```

**Web UI 模式：**
```bash
streamlit run code/C10/web_app.py
```

## 项目结构

```
Agentic-RAG/
├── config.env                     # 环境变量配置
├── CSV_DATA/                      # 临床 QA 原始数据
│   ├── Andriatria_男科/
│   ├── IM_内科/
│   ├── OAGD_妇产科/
│   ├── Oncology_肿瘤科/
│   ├── Pediatric_儿科/
│   └── Surgical_外科/
├── data/
│   └── C10/                       # Neo4j 数据 & 疾病词典
│       ├── docker-compose.yml
│       ├── disease_dict.xlsx      # 疾病词典
│       └── 典型病历.xlsx           # 示例病历数据
├── code/
│   ├── requirements.txt           # Python 依赖
│   ├── docker-compose.yml         # Milvus 容器编排
│   ├── m3_download.py             # 嵌入模型下载脚本
│   ├── models/                    # 本地嵌入模型权重
│   └── C10/
│       ├── main.py                # 系统入口 & 协调器
│       ├── web_app.py             # Streamlit 交互界面
│       ├── config.py              # 全局配置（数据类）
│       ├── clean_milvus.py        # 清理 Milvus 集合工具
│       └── rag_modules/
│           ├── graph_data_preparation.py   # CSV 导入 & 图谱构建
│           ├── milvus_index_construction.py # 向量索引构建
│           ├── graph_indexing.py           # 内存图索引（KV 存储）
│           ├── hybrid_retrieval.py          # 混合检索（向量 + BM25）
│           ├── graph_rag_retrieval.py       # 图谱检索（多跳/子图/路径）
│           ├── intelligent_query_router.py  # LLM 查询路由引擎
│           └── generation_integration.py    # DeepSeek 答案生成
└── README.md
```

## 核心模块说明

### GraphDataPreparationModule
从 CSV 加载各科室 QA 数据，利用疾病词典进行实体匹配，构建 Neo4j 知识图谱（Department → Consultation → Disease），并生成 LangChain Document 对象供向量索引使用。

### MilvusIndexConstructionModule
使用 BGE 系列模型对文档进行嵌入编码，管理 Milvus Collection 的创建、索引构建（HNSW）和插入操作，支持按科室过滤。

### HybridRetrievalModule
双层检索：实体级关键词匹配 + 主题级关系匹配（基于内存图索引），结合 BM25 与 Milvus 向量检索，通过轮巡策略融合结果。

### GraphRAGRetrieval
纯图谱检索模块，理解查询意图（实体关系、多跳推理、子图提取、路径发现），执行 Cypher 查询进行多跳遍历，构建推理链上下文。

### IntelligentQueryRouter
核心路由引擎——使用 DeepSeek 分析查询的复杂度和关系密度，决定使用混合检索、GraphRAG 还是融合策略，失败时降级为规则判断。

### GenerationIntegrationModule
构建医疗专用提示词，融合检索结果后调用 DeepSeek API 生成答案，支持流式输出、引用溯源与医疗免责声明。

## 配置说明

主要配置集中在 `code/C10/config.py`：

| 参数             | 说明                               | 默认值                        |
| ---------------- | ---------------------------------- | ----------------------------- |
| `NEO4J_URI`      | Neo4j 连接地址                     | `bolt://localhost:7687`       |
| `NEO4J_USER`     | Neo4j 用户名                       | `neo4j`                       |
| `NEO4J_PASSWORD` | Neo4j 密码                         | `password`                    |
| `MILVUS_HOST`    | Milvus 服务地址                    | `localhost`                   |
| `MILVUS_PORT`    | Milvus 端口                        | `19530`                       |
| `EMBEDDING_MODEL`| 嵌入模型路径或名称                 | `bge-base-zh-v1.5`            |
| `TOP_K`          | 检索返回 Top-K 文档数              | `5`                           |
| `ROUTER_MODEL`   | 路由 / 生成使用的 LLM 模型名称     | `deepseek-chat`               |

## License

MIT
