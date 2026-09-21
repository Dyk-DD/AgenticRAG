# 智能问答系统 - 前端

基于 React + Vite + TypeScript 的智能问答系统前端，通过 SSE 流式通信与本仓库根目录下的后端 API 交互。

## 技术栈

- React 19 + TypeScript
- Vite 8（开发/构建）
- Zustand（状态管理）
- React Router 7（路由）
- SSE（服务端推送流式响应）

## 功能

- **个人主页**：公开访问的作品集门户（`/`），问答系统作为其中一个作品卡片进入。新增作品只需往 `pages/HomePage.tsx` 的 `CARDS` 数组里加一项
- **对话问答**：流式输出，支持医疗症状、药物禁忌、临床推理等查询；生成中途可点「停止」中断（`stopStreaming`）
- **推理过程面板**：把后端智能路由的决策依据、复杂度、关系密集度收进一个默认折叠的面板，点开才展开
- **账号**：邮箱 + 密码注册登录，注册需邮箱验证码；忘记密码可用验证码自助重置
- **会话管理**：新建/切换/置顶/重命名/删除对话会话，都收在每个会话行右侧的「⋯」菜单里；会话名默认取首条提问
- **分享会话**：从「⋯」菜单里挑一轮或多轮问答，生成一条**完全公开**的链接（`/s/:token`）—— 拿到链接的人不需要访问密码也不需要账号
- **响应式**：侧边栏可折叠（宽屏偏好存 `localStorage`）；≤900px 转为抽屉 + 遮罩

侧栏被收起时（宽屏折叠 / 窄屏抽屉关闭），主区左上角浮出 `☰` / `＋` / `←` 三个按钮：这三个动作在侧栏里都够不着，而侧栏此时是 `inert` 的。`＋` 与 `←` 是后补的 —— 原先折叠态既无法新建会话，品牌字标又不再可点，等于没有回主页的路。

「返回主页」在侧栏品牌行左侧，**品牌字标本身不是链接**：它挂在最显眼的位置，误点就离开应用，而「离开应用」是个有后果的动作，不该由一个看起来只是标题的元素承担。

## 样式

全部样式都在 `src/index.css` 一个文件里，没有 CSS 框架、CSS Modules 或图标库。

**三套调色板各管各的界面，靠作用域隔离**：

- 公开主页在 `.home-page` 上声明一套 `--home-*`（暖白纸张 + 衬线标题），
- 对话界面在 `.app-layout` 上声明一套 `--chat-*`（浅色，对齐 DeepSeek 网页端），
- 公开分享页在 `.share-page` 上**再声明一份同样的 `--chat-*`**，

三者都**不从 `:root` 改色**，只把全局 `--primary` 当输入用。之所以能这么切：`ChatGate` 在未登录时渲染 `LoginPage` / `AccountGate` **代替** `<Outlet/>`，`ChatLayout` 根本不挂载，所以 `.app-layout` 天然把登录页与账号页隔离在外。

分享页要单独挂一份而不是复用 `.app-layout`：它是独立路由、不经过 `ChatGate`，与 `.app-layout` 不在同一棵 DOM 上，继承不到；而给它的外层套 `.app-layout` 会连带继承 `display:flex; height:100dvh`，分享页要的是自己的滚动容器和内容列。为此 `.app-layout, .share-page` 那条规则里**只放变量声明**，布局属性另立在 `.app-layout` 里 —— 新增一个作用域根时不会被顺带套上布局。

新增断点时注意：文件里三个 `@media` 块（900 / 640 / 720px）**各管一个界面**，块内只允许出现该界面前缀的选择器（`.chat-*` / `.sidebar-*` / `.home-*` …）。混进无作用域的选择器会同时波及另外两个界面。900px 块里那条 `padding-top: 64px` 只给 `.chat-messages` —— 那是给左上角浮动按钮让位的，分享页没有那组按钮。

**`--chat-*` 里的 `--chat-accent` 与 `--chat-accent-strong` 是两个值不是笔误**：DeepSeek 品牌蓝 `#4d6bfe` 对白字只有 4.33:1，载不了小字，但作填充/图标/边框（非文本，门槛 3:1）合格；凡是要承载文字的地方一律用 `-strong`（`#3b5bdb`，双向 5.67:1）。

## 账号

两道门：全局访问密码（`LoginPage`）挡在所有 `/api` 前面，账号（`AccountGate`）叠在它之后。

- **注册需要邮箱验证码**（`POST /api/accounts/email-code`，60 秒冷却 + 每邮箱每日上限）。验证码通过之前不会创建账号行，所以没人能拿别人的邮箱占坑。后端要在 `.env` 里配好「邮件验证码」那一节，否则注册与重置会返回 503 —— **登录不受影响**，它是唯一不需要发信的账号操作
- **密码可以自助重置**：`忘记密码？` → 邮箱验证码 → 新密码 → 回登录页（刻意不发新令牌，让用户自己登一次顺便证明新密码能用）。老「患者 ID + 访问码」身份的迁移/找回仍然只能请管理员跑后端仓库的 `scripts/migrate_sessions_to_account.py`
- **重设密码不能让已登录的设备下线**：令牌是无状态 HMAC、30 天有效，后端没有吊销名单。界面上照实说了这一点，别把它改写成「改密码就能把别人踢出去」
- 仍然不做 RFC 5322 校验、不查 MX，只拦手误。邮件头注入（`To:` 里的换行）另有一道守卫，在 `app/mailer.py` 里
- 账号接口的 403 表示**密码错误或验证码错误**，不是令牌失效，所以走 `client.ts` 的 `ensureAccountOk`（只有 401 才清身份并重载）。若误用 `ensureOk`，打错一次密码就会被踢出全局门禁，且填好的表单内容一并丢失。`npm run build` 只跑 `tsc`，eslint 不在构建路径里 —— 这条不变量只靠注释和评审活着
- 验证码只留在组件 state 里，**绝不写进 localStorage**；`code` 步骤也不做持久化，理由写在 `AccountGate.tsx` 的注释里

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器（默认 http://localhost:5173）
npm run dev
```

开发时 Vite 会将 `/api` 请求代理到后端 `http://localhost:8000`。

## 生产构建

```bash
npm run build
```

构建产物输出到 `dist/` 目录。

## 部署

**`dist/` 是活挂载，`npm run build` 就是发布动作。**

`dist/` 被 `docker-compose.yml` 以只读方式挂进 api 容器的 `/app/static`，由 FastAPI 同源提供（前端与 API 同一个 origin，因此没有 CORS，也不需要配置后端地址）。根路径与客户端路由的 fallback 由后端 `app/api.py` 的 catch-all 路由负责，所以 `/chat`、`/s/<token>` 硬刷新不会 404 —— 后者对分享链接尤其要紧：收链接的人多半就是直接粘进地址栏打开的，没有 SPA fallback 的话第一条命就没了。

因此构建流程是：

```bash
npm run build      # 产物直接写进 dist/，线上立即生效
```

⚠️ 迭代过程中不要跑 `npm run build` —— 那等于把半成品发布到线上。只做类型检查请用 `npx tsc -b`（`noEmit` 已开，不写盘）。

前端不内联任何密钥：所有请求都走**同源相对路径**，前端没有任何后端地址配置项。
用到的环境量只有两个 —— `import.meta.env.DEV`（决定走不走 Vite 代理），以及
`VITE_ACADEMIC_URL`（主页上站外应用那张卡片）。后者写在不进版本控制的 `.env`
里；不设它，那张卡片就不出现，主站功能不受影响。

## 项目结构

```text
src/
├── api/
│   ├── client.ts          # API 客户端（REST + SSE）
│   └── config.ts          # 令牌与患者身份的 localStorage 读写
├── store/chatStore.ts     # Zustand 状态管理
├── lib/routing.ts         # 策略中文名与百分比格式化（三个界面共用一份）
├── types/index.ts         # TypeScript 类型定义
├── components/
│   ├── ChatGate.tsx       # /chat 子树的两道门（访问密码 + 账号）
│   ├── LoginPage.tsx      # 访问密码登录页
│   ├── AccountGate.tsx    # 账号注册/登录页
│   ├── ChatLayout.tsx     # 侧边栏（可折叠）+ 主布局
│   ├── SessionItem.tsx    # 单条会话：标题 + 「⋯」菜单（置顶/重命名/分享/删除）
│   ├── ShareDialog.tsx    # 选轮次生成公开链接（渲染在 .main-content 里）
│   ├── ChatMessage.tsx    # 单条消息（助手去气泡，用户右侧浅色气泡）
│   ├── ChatInput.tsx      # 输入框（含停止生成按钮）
│   └── RoutingCard.tsx    # 「推理过程」折叠面板
└── pages/
    ├── HomePage.tsx       # 公开个人主页（/）
    ├── ChatPage.tsx       # 对话页
    └── SharePage.tsx      # 公开分享页（/s/:token）
```

`SessionItem.tsx` 把原先的 `RenameableTitle.tsx` 折了进来。那个组件的存在理由是「侧栏与历史记录页共用」，历史页删掉后它只剩一个消费者，同时「重命名」从行内铅笔移进菜单后编辑态必须能被菜单触发 —— 两个理由都指向合并。合并时原样保留了它的三条硬约束：触发编辑要 `stopPropagation`、编辑态输入的 `onClick` **和** `onKeyDown` 都要 `stopPropagation`、**绝不加 `onBlur` 提交**（会和 Enter 抢触发导致双提交）。

`ShareDialog` 渲染在 `.main-content` 里而不是侧栏里，这不是风格选择：它是 `inset:0` 的全屏遮罩，而窄屏抽屉态的 `.sidebar` 带 `transform`，会成为 `fixed` 后代的包含块 —— 挂在侧栏里会被压进 280px 的抽屉。同理，会话行的「⋯」菜单用 `position:absolute` 且包含块是 `.sidebar`（它带 `position: relative`），**绝不能给 `.session-item` 加 `position: relative`**：那会把包含块拽回滚动容器 `.session-list` 内部，菜单当场被裁。理由写在 `index.css` 里那两条注释上。

`lib/routing.ts` 独立成 `.ts` 而不是并进组件文件：本仓库 `react-refresh/only-export-components` 是 error，组件文件里导出非组件会让该文件的 Fast Refresh 整体失效。

## 路由

| 路径 | 组件 | 鉴权 |
| --- | --- | --- |
| `/` | `HomePage` | 公开 |
| `/chat` | `ChatLayout` → `ChatPage` | `ChatGate` |
| `/s/:token` | `SharePage` | **公开**（拿到链接即可看） |
| `/history` | 重定向到 `/chat` | 无（在 `ChatGate` 之外） |
| `*` | 重定向到 `/` | 无 |

`/s/:token` 的位置是承重的：它必须在 `ChatGate` **外面**（放里面会让访问者撞上访问密码门），也必须在 `path="*"` **之前**（否则被兜底重定向吃掉，表现成「分享链接打开后跳到主页」）。两条都是「看起来只是顺序」但改错了功能就没了的那类约束，注释写在 `App.tsx` 里。

`/history` 留着是因为它当初对外发布过，是旧书签的入口；`/chat/history` 一并失效 —— 它只是 `/history` 的重定向目标，没有单独流出去过，不必再留一层。

`ChatGate` 把未登录状态**原地渲染**成登录页，而不是重定向到 `/login`。这不是风格选择：`api/client.ts` 在收到 401/403 时会 `clearToken()` + `window.location.reload()`，刷新后仍停在 `/chat`，守卫必须能在同一 URL 上接住未登录态；换成重定向会丢掉用户原本要去的深链。
