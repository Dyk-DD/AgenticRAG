import {
  getToken,
  getPatientToken,
  clearToken,
  clearPatientId,
} from './config';
import type { Session, SessionDetail, ShareTurn } from '../types';

// 所有请求都走同源相对路径：生产环境 SPA 由 FastAPI 同源提供（dist 只读挂载进
// api 容器），DEV 下 /api 由 vite.config.ts 的 proxy 转发到 127.0.0.1:8000。
// 前端没有、也不再有「后端地址」这个配置项 —— 曾经的 api_base 覆盖机制与
// 「⚙ 后端连接」弹窗一起删掉了（那个覆盖在 DEV 下从未生效过，因为
// getApiBase() 无条件返回空串；在生产下也没有可配置的东西，前后端同源）。

function hdrs(extra?: Record<string, string>): Record<string, string> {
  const token = getToken();
  const patientToken = getPatientToken();
  const base: Record<string, string> = {};
  if (token) base['Authorization'] = `Bearer ${token}`;
  // 患者身份用服务端签发的令牌，不再发送可随意伪造的患者 ID
  if (patientToken) base['x-patient-token'] = patientToken;
  return { ...base, ...extra };
}

/**
 * 带状态码的接口错误。只在 UI 需要区分「等 N 秒再来」这类可重试失败时才用它，
 * 其余地方继续只读 message（它是 Error 的子类，`e instanceof Error` 照样成立）。
 */
export class ApiError extends Error {
  readonly status: number;
  /** 仅当服务端给了 Retry-After 时有值（秒）。 */
  readonly retryAfter?: number;

  constructor(message: string, status: number, retryAfter?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = '';
  try {
    const body = await res.json();
    detail = body?.detail || '';
  } catch {
    // 响应体不是 JSON，退回状态码
  }
  // 只在 429 上读这个头，而且只信正数。后端也只有「冷却未过」那一种 429 会带它
  // （配额用尽的 429 刻意不带），所以接上倒计时不会把「今天没戏了」伪装成
  // 「等 60 秒就好」。
  const raw = res.status === 429 ? Number(res.headers.get('Retry-After')) : NaN;
  return new ApiError(
    detail || `HTTP ${res.status}`,
    res.status,
    Number.isFinite(raw) && raw > 0 ? raw : undefined,
  );
}

/** 统一的响应校验：401/403 说明令牌缺失、过期或无效，清空本地身份回到登录页。 */
async function ensureOk(res: Response): Promise<void> {
  if (res.ok) return;

  if (res.status === 401 || res.status === 403) {
    clearToken();
    clearPatientId();
    window.location.reload();
  }

  throw await errorFrom(res);
}

/**
 * 账号接口专用的响应校验：只有 401 才清本地身份并整页重载，403 原样抛出。
 *
 * 不能复用 ensureOk —— 它把 403 也当成「令牌失效」。账号登录/注册时密码打错
 * 返回的正是 403，走 ensureOk 就会清掉全局访问令牌并 reload：打错一次密码不但
 * 被踢出全局门禁，填好的表单也一起没了。
 *
 * **验证码错误同样是 403，理由相同**：填错一个码不该清身份、不该整页重载、
 * 更不该把用户填好的邮箱和密码一起丢掉。后端为此专门用 403 而不是 401
 * （见 api.py 的 _CODE_ERRORS），改这里之前先看那边。
 *
 * 所有账号接口都必须走本函数。这不是风格问题：npm run build 只跑 tsc，eslint
 * 不在构建路径里，而 tsc 看不见「用错了校验函数」—— 它只能靠这条注释和评审。
 *
 * 401 仍然要重载，但那指的是全局访问令牌失效（这些端点只过全局中间件，
 * 400/403/404/409/429/502/503 都是业务错误），那确实该回登录页。
 */
async function ensureAccountOk(res: Response): Promise<void> {
  if (res.ok) return;

  if (res.status === 401) {
    clearToken();
    clearPatientId();
    window.location.reload();
  }

  throw await errorFrom(res);
}

export type CodePurpose = 'register' | 'reset';

/** 请求发送邮箱验证码。响应里只有 sent/expires_in/resend_after，永远没有码本身。 */
export async function sendEmailCode(email: string, purpose: CodePurpose) {
  const res = await fetch('/api/accounts/email-code', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ email, purpose }),
  });
  await ensureAccountOk(res);
  return res.json() as Promise<{ sent: boolean; expires_in: number; resend_after: number }>;
}

// code 是必填参数而不是可选项：AccountGate 是唯一调用方，让 tsc 在漏改调用点时
// 直接报错，而不是留到运行时变成一个 400。
export async function registerAccount(
  email: string,
  password: string,
  name: string,
  code: string,
) {
  const res = await fetch('/api/accounts/register', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ email, password, name, code }),
  });
  await ensureAccountOk(res);
  return res.json();
}

export async function loginAccount(email: string, password: string) {
  const res = await fetch('/api/accounts/login', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ email, password }),
  });
  await ensureAccountOk(res);
  return res.json();
}

/** 验证码通过后重设密码。刻意不返回令牌，成功后就回登录页让用户自己登一次。 */
export async function resetPassword(email: string, code: string, newPassword: string) {
  const res = await fetch('/api/accounts/password-reset', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ email, code, new_password: newPassword }),
  });
  await ensureAccountOk(res);
  return res.json() as Promise<{ reset: boolean }>;
}

export async function chatNonStream(question: string) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ question, stream: false }),
  });
  await ensureOk(res);
  return res.json();
}

export function chatStream(
  question: string,
  onRouting: (data: any) => void,
  onToken: (text: string) => void,
  onDone: (data: any) => void,
  onError: (msg: string) => void,
): AbortController {
  const controller = new AbortController();

  fetch('/api/chat/stream', {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ question, stream: true }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        // 流式响应无法复用 ensureOk 的 throw 语义，这里单独处理鉴权失效
        if (response.status === 401 || response.status === 403) {
          clearToken();
          clearPatientId();
          window.location.reload();
          return;
        }
        let detail = '';
        try {
          const body = await response.json();
          detail = body?.detail || '';
        } catch {
          // 响应体不是 JSON，退回状态码
        }
        onError(detail || `HTTP ${response.status}`);
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) { onError('No response body'); return; }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let eventType = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (eventType === 'routing') onRouting(parsed);
              else if (eventType === 'token') onToken(parsed.text);
              else if (eventType === 'done') onDone(parsed);
              else if (eventType === 'error') onError(parsed.message);
            } catch {
              // skip unparseable lines
            }
            eventType = '';
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err.message);
    });

  return controller;
}

export async function fetchSessions() {
  const res = await fetch('/api/sessions', { headers: hdrs() });
  await ensureOk(res);
  return res.json() as Promise<{ sessions: Session[] }>;
}

/**
 * 会话详情。**必须带返回类型**，不能像别处那样只写 res.json()：分享对话框靠
 * turns[].turn_index 挑轮次，返回 any 会让「后端真的透出了 turn_index」这件事
 * 失去编译期保障 —— 后端哪天漏掉它，只会在运行时表现为 undefined 传进
 * ShareCreate，而那是在**公开页面**上错位。
 */
export async function fetchSessionDetail(id: string) {
  const res = await fetch(`/api/sessions/${id}`, { headers: hdrs() });
  await ensureOk(res);
  return res.json() as Promise<SessionDetail>;
}

export async function deleteSession(id: string) {
  const res = await fetch(`/api/sessions/${id}`, { method: 'DELETE', headers: hdrs() });
  await ensureOk(res);
  return res.json();
}

export async function createSession() {
  const res = await fetch('/api/sessions', { method: 'POST', headers: hdrs() });
  await ensureOk(res);
  return res.json();
}

export async function activateSession(sessionId: string) {
  const res = await fetch(`/api/sessions/${sessionId}/activate`, { method: 'PUT', headers: hdrs() });
  await ensureOk(res);
  return res.json();
}

/**
 * 改会话标题。用 ensureOk 而不是 ensureAccountOk：这条路径上没有「密码打错」
 * 这种 403，401 要么是全局令牌失效、要么是身份令牌失效，两种都该清干净回门禁。
 */
export async function renameSession(id: string, title: string) {
  const res = await fetch(`/api/sessions/${id}`, {
    method: 'PUT',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  });
  await ensureOk(res);
  return res.json();
}

/**
 * 置顶 / 取消置顶。**不带请求体** —— 后端用 PUT 与 DELETE 两个方法表达意图，
 * 没有可以拼错的字段（详见 app/api.py 里 pin_session 的说明）。
 */
export async function pinSession(id: string, pinned: boolean) {
  const res = await fetch(`/api/sessions/${id}/pin`, {
    method: pinned ? 'PUT' : 'DELETE',
    headers: hdrs(),
  });
  await ensureOk(res);
  return res.json();
}

/**
 * 创建分享链接。turns 传的是**真实 turn_index**，不是数组下标 ——
 * 下标是隐式契约，任何一次前端过滤都会让它静默错位，把 A 轮的答案公开挂在
 * B 轮的问题下面（详见 app/api.py 里 create_share 的说明）。
 */
export async function createShare(sessionId: string, turns: number[]) {
  const res = await fetch(`/api/sessions/${sessionId}/share`, {
    method: 'POST',
    headers: hdrs({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ turns }),
  });
  await ensureOk(res);
  return res.json() as Promise<{
    token: string;
    path: string;
    title: string;
    created_at: string;
    turn_count: number;
  }>;
}

export async function revokeShare(token: string) {
  // 注意是 /api/shares/（复数）：公开前缀 /api/share/ 是 GET 绑定的，
  // 这个 DELETE 撤不到那条豁免，仍走全局访问令牌与患者身份两道门。
  const res = await fetch(`/api/shares/${token}`, { method: 'DELETE', headers: hdrs() });
  await ensureOk(res);
  return res.json();
}

/**
 * 读取一条分享。**公开接口，故意不走 ensureOk。**
 *
 * 分享页是给拿到链接的陌生人看的，他们身上既没有访问令牌也没有患者身份，
 * 请求必然带着 401/403 —— 走 ensureOk 会触发 clearToken() + clearPatientId()
 * + window.location.reload()，把一个只是来读公开链接的人整页刷新一遍，
 * 而且反复刷新（每次加载都再失败一次）。这是 ensureAccountOk 那条教训的
 * 同一个形状：校验函数选错不会有编译错误，只会在运行时做出破坏性的事。
 */
export async function fetchShare(token: string) {
  const res = await fetch(`/api/share/${token}`);
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<{
    title: string;
    created_at: string;
    turns: ShareTurn[];
  }>;
}

export async function fetchStats() {
  const res = await fetch('/api/stats', { headers: hdrs() });
  await ensureOk(res);
  return res.json();
}
