# Agentic-RAG：图谱增强检索智能问答系统

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
| API 层     | FastAPI + SSE 流式                                       |
| 前端       | React 19 + TypeScript + Vite                             |
| 部署       | Docker Compose 全栈编排 + Cloudflare Tunnel 固定域名      |
| 访问控制   | HMAC-SHA256 签名令牌 + PBKDF2 访问码 + 按患者会话隔离      |

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- NVIDIA GPU（可选，但强烈建议）

**关于 GPU**：嵌入模型 BGE 的推理速度决定了访客要等多久才看到第一个字。
图谱检索会对 150 个候选文档做一次语义重排（`graph_rag_retrieval.py` 的
`max_rerank_candidates`），这一段实测：

| 设备 | `graph_rag_search` | 端到端首 token |
| --- | --- | --- |
| CPU（容器，仅 torch CPU 版） | 50.6s | 74.3s |
| RTX 3060（首次，含 CUDA 上下文初始化） | 14.8s | 20.2s |
| RTX 3060（后续） | 4.0s | 17.6s |

CPU 那一列会顶到 Cloudflare 100 秒的源站超时上限，所以有独显时建议走 GPU。

镜像默认装 CUDA 版 torch。**没有 N 卡的机器不用改任何东西**——CUDA wheel
自带完整 CPU 后端，会自动回落，只是慢。想省体积（实测 6.74GB → 2.04GB）：

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

用 GPU 就要保留 `docker-compose.yml` 里 `api` 服务的 `deploy.resources` 段；
没有 N 卡则**必须删掉它**，否则容器起不来。

构建默认走清华 PyPI 镜像（官方源 + `pypi.nvidia.com` 在国内实测会断流）。
要换回官方源：

```bash
docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
```

用了 GPU 就要在 `docker-compose.yml` 的 `api` 服务里保留 `deploy.resources`
那段；没有 N 卡则删掉它（否则容器起不来）。

### 1. 克隆与配置

```bash
git clone <repo-url>
cd Agentic-RAG
```

复制环境变量模板并填写：
```bash
cp .env.example .env
```

必须填写的三项：
```env
DEEPSEEK_API_KEY=sk-your-key-here   # DeepSeek 控制台申请
ACCESS_PASSWORD=                    # 访问系统的全局密码
NEO4J_PASSWORD=                     # Neo4j 密码，与 docker compose 共用
```

`SECRET_KEY` 用于签发访问令牌与患者身份令牌，留空时会从 `ACCESS_PASSWORD`
派生；**公网部署务必单独设置**，生成方式：
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

> `ACCESS_PASSWORD` 留空时后端默认拒绝全部受保护接口（fail closed），
> 本机调试可临时设置 `ALLOW_NO_AUTH=1`，切勿在公网启用。

#### 模型选型：必须用非推理模型

`LLM_MODEL` 默认 `deepseek-chat`，**不要换成推理模型**（`deepseek-reasoner`、
`deepseek-v4-flash` 一类）。推理模型先输出 `reasoning_content`（思考）再输出正文，
二者共用同一个 `max_tokens` 预算；而本仓库没有任何一处读 `reasoning_content`，
于是预算被思考吃光时正文变成空字符串，且**不抛任何异常**。

代价是整条流水线静默降级，且四个症状看起来互不相关，很难一眼归因到模型上：

| 症状 | 实际原因 |
|------|----------|
| 回答只有一句免责声明、正文为空 | 生成时思考吃光了 `max_tokens` |
| 日志出现 `未能提取有效 JSON，降级为规则匹配` | 路由的 JSON 输出被思考挤掉 |
| 日志出现 `医学关键词提取失败` | 关键词抽取的 JSON 输出被思考挤掉 |
| 实体级 / 临床主题级检索恒返回 0 条 | 上一条导致双层检索没有输入 |

最后一条尤其隐蔽：**「Agentic」的双层检索会整个失效，只剩向量层单路在工作**，
而系统表面完全正常。实测 `deepseek-chat` 思考 0 字、首字延迟 ~0.8s，
而推理模型在同一批问题上是 ~8s 首字且约 1/5 的正文为空。

代码侧已加了两道防线（生成器空正文直接报错、评测 judge 不静默兜底），
但模型选型仍是根因，换模型前请先读 `app/config.py` 中 `llm_model` 处的注释。

### 2. 启动服务

```bash
cd frontend && npm run build && cd ..   # 首次需要：构建 SPA 产物
docker compose up -d
```

一条命令起全栈：Milvus、etcd、MinIO、Neo4j，以及后端 API。API 容器同时托管前端
页面和接口，本机直接打开 `http://localhost:8000` 即可使用，无需另起 dev server。

首次启动要加载嵌入模型并检查图谱同步，需要几分钟。想等真正就绪再加 `--wait`。

> 只想跑数据存储、后端在宿主机上原生开发时：
> `docker compose up -d etcd minio milvus neo4j`

数据库端口：Milvus `19530`，Neo4j Bolt `8687`，Neo4j HTTP `8474`

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

**Web UI 模式（FastAPI + React）：**

容器化部署下，页面和接口都由 `api` 容器提供，直接开 `http://localhost:8000` 即可，
不需要单独启动前端。

要在宿主机上原生调试后端：
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
此时 `app/api.py` 会自动回落到 `<repo>/frontend/dist` 提供页面；
若该目录不存在，接口仍可用，但静态页面返回 404 并提示先构建前端。

前端开发模式（带热更新，API 自动代理到 8000）：
```bash
cd frontend && npm run dev
```

## 公网部署：Cloudflare Tunnel + 固定域名

前端与后端跑在同一组容器里，由一条 Cloudflare Tunnel 暴露，共用一个 HTTPS 域名。
同源意味着没有跨域、没有 mixed content，访客打开一个链接就能用，无需任何配置。

```
浏览器 ──HTTPS──▶ Cloudflare 边缘 ──隧道──▶ cloudflared 容器
                                                │ http://api:8000
                                                ▼
                                        api 容器（FastAPI + SPA 静态资源）
                                          ├── bolt://neo4j:7687
                                          └── milvus:19530
```

宿主机不开放任何入站端口——隧道是纯出站连接。

> **代价要说清楚**：后端跑在个人电脑上，机器关机或休眠时站点就打不开。
> 这是「本地后端 + 隧道」这种形态的固有代价，不是配置问题。

### 1. 域名与隧道

1. 买一个域名（`.top` / `.xyz` 首年约 ¥10–30），在注册商处把 NS 改成 Cloudflare 分配的两个地址。
   **改完不要用公共 DNS 判断是否生效**：`.com` 的顶级域委派有最长 48 小时的刷新周期，
   `nslookup -type=NS 你的域名 8.8.8.8` 可能一整天都还返回注册商的旧 NS。注册局侧是否已接受，
   查 whois 更准（`whois 你的域名 | grep -i "name server"`）。
2. 登录 [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → Networks → Tunnels → **Create a tunnel**，选 Cloudflared，命名如 `agentic-rag`
3. 复制生成的 **token**（只显示一次）
4. 在该隧道的 **Public Hostnames** 里添加一条：Subdomain 填 `www`（随意，本项目用
   `www.你的域名`），Domain 选你的域名，Service 类型选 `HTTP`，
   URL 填 **`api:8000`**
5. 把 token 写进 `.env` 的 `TUNNEL_TOKEN=`

第 4 步的 URL 必须填 `api:8000` —— 这是 compose 网络内的服务名，**不是 `localhost`**。
Cloudflare 会自动为 `www.你的域名` 建一条指向隧道的 CNAME，DNS 无需手动配置。

> **填错这里会得到什么**：写 `localhost:8000` 的话，连接器容器里的 `localhost`
> 是它自己的回环，不是 api 容器 —— 日志里会是
> `Unable to reach the origin service ... dial tcp [::1]:8000: connect: connection refused`。
> 改完记得在控制台点保存，改完可以重启连接器核对实际生效的配置：
> ```bash
> docker compose logs cloudflared | grep -o 'ingress.*http_status'
> ```
> 输出里的 `"service"` 字段就是真正生效的源站地址。

> **宿主机上不要再跑一个 cloudflared**。用同一个 token 在宿主机装过服务
> （`cloudflared service install`）的话，它会和容器里的连接器同时挂到同一条隧道上，
> Cloudflare 把请求轮流分给两者。宿主机那个解析 `localhost:8000` 是通的（走 api 的
> 已发布端口），容器那个不通 —— 表现出来就是**大约一半请求 502、一半正常**，
> 极难排查。装过就停掉：
> ```powershell
> sc.exe stop Cloudflared
> sc.exe config Cloudflared start= disabled
> ```

> 入口规则存在 Cloudflare 控制台而不是仓库里，这是令牌模式（dashboard 托管）
> 换取"不必在容器里管理 cert.pem 与长期凭据文件"的代价。

### 2. 启动

```bash
cd frontend && npm run build && cd ..   # 构建 SPA 产物（容器只读挂载它）
docker compose up -d --wait             # --wait 会等到 /api/health 真正就绪
```

`/api/health` 在 RAG 未就绪时也返回 200，但 body 是 `{"status":"initializing"}`。
健康检查是看 body 而不是状态码的，所以 `--wait` 期间的等待是真实的——首次启动
要加载嵌入模型并核对图谱同步，几分钟属正常。

### 3. Cloudflare 侧需要关掉的开关

| 开关 | 位置 | 为什么 |
| --- | --- | --- |
| Rocket Loader | Speed → Optimization | 会延迟加载 `type="module"` 的 bundle，页面直接白屏 |
| Auto Minify | Speed → Optimization | 可能破坏已压缩的 JS 产物 |
| Bot Fight Mode / Under Attack | Security | 会往响应里注入 JS 挑战，SPA 的 `fetch` 无法通过，前端拿到的是 403 HTML 而不是 JSON |

不要给 `/api/*` 配 "Cache Everything" 规则。SSE 流式接口 `/api/chat/stream`
依赖 `text/event-stream` 不被缓冲或压缩，Cloudflare 默认不会动这个 content type。

实测过这条：开着机器人防护时，`User-Agent: Python-urllib/3.12` 会被直接挡下
（纯文本 403，无 `cf-mitigated` 挑战头），而 Chrome 与 curl 的 UA 均正常放行。
真实浏览器因此不受影响，但任何非浏览器客户端（监控探针、脚本化的冒烟测试、
`curl` 之外的 HTTP 库）都会拿到 403 —— 用 Python 写验证脚本时要自己带上浏览器 UA。

### 4. 数据在哪里

SQLite（患者、会话、问答记录）存在 Docker 命名卷 `rag_state` 里，不落宿主机目录。
这是刻意的：SQLite 的 WAL 模式依赖共享内存与 POSIX 建议锁，跑在 Docker Desktop 的
Windows 挂载上是已知的不可靠组合，并发下会 `database is locked`，甚至静默损坏。

备份出来：

```bash
docker run --rm -v agentic-rag_rag_state:/src -v "$PWD/data:/dst" alpine \
  cp -av /src/qa_history.db /dst/
```

### 5. 本机验证（不走隧道）

容器内已经同时提供前端和 API，直接开 `http://localhost:8000` 走完整流程即可。

#### 5.1 先发布端口

`api` 容器默认不发布端口。在仓库根目录新建 `docker-compose.override.yml`：

```yaml
services:
  api:
    ports:
      - "127.0.0.1:8000:8000"
```

`docker compose` 会自动读取并叠加这个文件。**改这里而不是改
`docker-compose.yml`**：override 文件已在 `.gitignore` 里，不进版本控制，
`git pull` 时不会冲突。

然后 `docker compose up -d api` 生效。注意这会**重建 api 容器**，容器内的嵌入
模型要重新加载，公网站点在此期间不可用；挑个没人用的时候做。
撤销：删掉该文件再 `docker compose up -d api`。

**不要绑到 `0.0.0.0`**——限流依赖 `cf-connecting-ip` 头，端口一旦对别的来源
可达，该头就能被伪造，等于绕过 `/api/auth` 的暴破防护。

#### 5.2 前端本地开发

前端产物是**只读挂载**（`./frontend/dist:/app/static:ro`），从不打进镜像——
所以改前端**永远不需要重建镜像**，也不必绕公网：

| 目的             | 命令                           | 生效方式                                    |
| ---------------- | ------------------------------ | ------------------------------------------- |
| 改设计（热更新） | `cd frontend && npm run dev`   | `http://localhost:5173`，保存即刷新         |
| 验证产物         | `cd frontend && npm run build` | dist 实时挂载，刷新页面即可，容器都不用重启 |

热更新模式依赖 5.1 那步：`app/api/config.ts` 在 DEV 下让 API 走相对路径，
由 Vite 代理转发到 `127.0.0.1:8000`，端口没发布就会得到 502。

在 VSCode 里 **Open Folder 直接打开 `frontend/`**，不要开仓库根目录——它有自己的
`package.json` / `tsconfig` / `eslint.config.js`，开根目录会让 TS 服务按后端那套
配置解析，飘一堆假报错。

#### 5.3 跑评测

`scripts/local_test.sh` 把「拷脚本进容器 → 跑评测 → 清测试数据」封成一条命令：

```bash
./scripts/local_test.sh                 # 导入检查 + 检索/路由评测(20条) + 清数据
./scripts/local_test.sh --full          # 加 generation/e2e + LLM Judge（慢、花钱）
./scripts/local_test.sh --smoke         # 只导入检查，不花 LLM 钱
./scripts/local_test.sh --sample 50     # 换采样条数
./scripts/local_test.sh --no-clean      # 跑完不清数据
./scripts/local_test.sh --clean-only    # 跳过检查与评测，只清数据
./scripts/local_test.sh --dry-run       # 只统计待删记录数，不做任何删除
```

输出实时流式，同时留档到 `scripts/.local_test_last.log`。评测会把患者/会话/问答
写进 SQLite，默认跑完自动清掉（只动 `qa_history.db`，不碰 Milvus 与 Neo4j）。

脚本内部处理了两件必须做对的事：`scripts/` 在 `.dockerignore` 里（镜像中没有）
所以要先 `docker cp` 进容器，且落点必须是 `/app/scripts/` —— `evaluate.py` 用
`__file__` 反推项目根，放 `/tmp` 会 import 不到 `app.*`；Git Bash 下还要
`MSYS_NO_PATHCONV=1`，否则容器路径会被改写成 Windows 路径。

> **测试集必须与知识库同源**，否则检索指标恒为 0：ground truth 的 node_id 由
> `md5(科室_标题_问_答)` 算出，拿别的科室的 CSV 当测试集一条都匹配不上。
> 用 `scripts/strategy_comparison/sample_testset.py` 采样、
> `label_routing_ground_truth.py` 标注路由 ground truth 生成。
> 直接跑公网 `https://www.yankandou.com` 做人工验收也可以，但只有这条路能出
> 可量化的指标。

## 项目结构

```
Agentic-RAG/
├── .env                  # API key、访问密码、隧道 token
├── .env.example          # 环境变量模板
├── requirements.txt      # Python 依赖
├── Dockerfile            # 后端 API 镜像（同时托管前端静态资源）
├── .dockerignore
├── docker-compose.yml    # 全部服务编排
│
├── app/                  # 主应用
│   ├── main.py           # 系统入口 & 协调器
│   ├── api.py            # FastAPI：REST + SSE，并托管前端页面
│   ├── security.py       # HMAC 签名令牌（访问令牌 + 患者身份令牌）
│   ├── qa_database.py    # SQLite 会话与问答持久化
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
| `llm_model` | `deepseek-chat` | 生成模型，**必须是非推理模型**（原因见「模型选型」一节） |
| `max_tokens` | `4096` | 生成上限，需调用方显式传入才生效 |
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
