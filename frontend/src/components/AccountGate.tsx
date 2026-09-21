import { useEffect, useState } from 'react';
import {
  setPatientId,
  setPatientToken,
  setPatientLabel,
  hasPatient,
} from '../api/config';
import {
  registerAccount,
  loginAccount,
  sendEmailCode,
  resetPassword,
  ApiError,
  type CodePurpose,
} from '../api/client';

interface Props {
  onReady: () => void;
}

/** 与后端 qa_database.MIN_PASSWORD_LEN 保持一致，本地先拦一道给出明确文案。 */
const MIN_PASSWORD_LEN = 8;
/** 与后端 api.py 的 _validate_code_or_400 一致。 */
const CODE_LEN = 6;

type Mode = 'login' | 'register' | 'reset';
/** credentials = 填邮箱/密码；code = 填收到的验证码。 */
type Step = 'credentials' | 'code';

const SUBTITLE: Record<Mode, string> = {
  login: '登录以继续',
  register: '注册一个新账号',
  reset: '用邮箱验证码重设密码',
};

/**
 * 账号门。登录、注册、忘记密码都在同一屏，用 mode + step 两个状态切换，
 * 不走路由 —— ChatGate 是原地渲染而不是重定向（原因见 ChatGate.tsx），
 * 而 client.ts 在 401 时会 clearToken() + reload，同 URL 上必须能接住。
 *
 * 注册与重置都要先发一封验证码邮件。发码与验码是两个请求，中间那一屏
 * （step === 'code'）必须保住用户已经填好的邮箱、昵称、密码，否则每次
 * 都要重敲一遍。
 *
 * 文案要诚实：密码**可以**通过邮箱验证码重置了（以前不行），但重设密码
 * **不能**把已经泄露出去的登录令牌赶下线 —— 令牌是无状态 HMAC、30 天有效，
 * 后端没有吊销名单。说成「改密码就能把别人踢下线」是错的。
 */
export default function AccountGate({ onReady }: Props) {
  const [mode, setMode] = useState<Mode>('login');
  const [step, setStep] = useState<Step>('credentials');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  // 验证码实际发往的地址。用它而不是 email 来展示，这样即使 email 后来被改，
  // 界面也不会声称「已发送至」一个从没收到过码的地址。
  const [sentTo, setSentTo] = useState('');
  const [cooldown, setCooldown] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (hasPatient()) onReady();
  }, []);

  // 重发倒计时。这是 src/ 里第一个 setInterval，所以两点写清楚：
  //   - 函数式更新 (c) => Math.max(0, c - 1) 让 interval 永远读不到闭包里的
  //     陈旧 cooldown，倒计时也不可能变成负数；
  //   - 返回的 cleanup 保证 onReady() 之后 ChatGate 把本组件换成 Outlet 时，
  //     不会有 setState 打在已卸载的组件上。
  // 依赖数组写 [cooldown] 意味着每秒拆建一次定时器，笨但显然正确；写成
  // [cooldown > 0] 能避免这点churn，但会触发 react-hooks/exhaustive-deps 的
  // 复杂表达式告警。
  //
  // 倒计时只是装饰：服务端的 429 才是权威，计时器归零后再点也可能被拒。
  useEffect(() => {
    if (cooldown <= 0) return;
    const id = window.setInterval(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => window.clearInterval(id);
  }, [cooldown]);

  const startCooldown = (seconds: unknown) => {
    // 用服务端下发的 resend_after，绝不在前端硬编码 60 —— 两边的冷却时间
    // 一旦各写各的，用户就会看到「按钮可点了但一发就 429」。
    const n = Number(seconds);
    setCooldown(Number.isFinite(n) && n > 0 ? n : 0);
  };

  const applyAccount = (data: { patient_id?: string; patient_token?: string; name?: string; email?: string }) => {
    // 没有令牌就进去，只会让每个请求拿到 401 → 清空身份 → reload → 弹回本页，
    // 变成一个看不见原因的刷新循环。停在这里说清楚。
    if (!data.patient_token) {
      setError('服务端未返回身份令牌，请重试');
      return false;
    }
    setPatientId(data.patient_id || '');
    setPatientToken(data.patient_token);
    setPatientLabel(data.name || data.email || email.trim());
    onReady();
    return true;
  };

  const requestCode = async () => {
    const mail = email.trim();
    if (!mail) return;
    const purpose: CodePurpose = mode === 'reset' ? 'reset' : 'register';
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const data = await sendEmailCode(mail, purpose);
      setSentTo(mail);
      startCooldown(data?.resend_after);
      setCode('');
      setStep('code');
    } catch (e) {
      setError(e instanceof Error ? e.message : '无法连接后端');
      // 冷却的 429 带着 Retry-After，把它接上倒计时。没有这条，刷新过页面的用户
      // 会拿到一个能点但每次都失败的按钮（state 被刷新清空，倒计时从 0 开始）。
      // 配额用尽的 429 不带这个头，于是正确地只显示文案、不装作等一会儿就好。
      if (e instanceof ApiError && e.status === 429 && e.retryAfter) {
        startCooldown(e.retryAfter);
      }
    } finally {
      setLoading(false);
    }
  };

  const submit = async () => {
    if (loading) return;
    const mail = email.trim();

    if (step === 'code') {
      if (code.length !== CODE_LEN) {
        setError(`请输入 ${CODE_LEN} 位验证码`);
        return;
      }
      if (mode !== 'login' && password.length < MIN_PASSWORD_LEN) {
        setError(`密码至少 ${MIN_PASSWORD_LEN} 位`);
        return;
      }
      setLoading(true);
      setError('');
      try {
        if (mode === 'reset') {
          await resetPassword(mail, code, password);
          // 重置刻意不签发令牌：回登录页让用户自己登一次，顺便证明新密码能用。
          setMode('login');
          setStep('credentials');
          setPassword('');
          setCode('');
          setCooldown(0);
          setNotice('密码已重置，请用新密码登录。');
          return;
        }
        applyAccount(await registerAccount(mail, password, name.trim(), code));
      } catch (e) {
        setError(e instanceof Error ? e.message : '无法连接后端');
      } finally {
        setLoading(false);
      }
      return;
    }

    // ── step === 'credentials' ──
    if (mode === 'login') {
      if (!mail || !password) return;
      setLoading(true);
      setError('');
      setNotice('');
      try {
        applyAccount(await loginAccount(mail, password));
      } catch (e) {
        setError(e instanceof Error ? e.message : '无法连接后端');
      } finally {
        setLoading(false);
      }
      return;
    }

    // 注册 / 重置的凭据屏：校验完就去要验证码
    if (!mail || !password) return;
    if (mode === 'register' && password.length < MIN_PASSWORD_LEN) {
      setError(`密码至少 ${MIN_PASSWORD_LEN} 位`);
      return;
    }
    await requestCode();
  };

  /** 切模式时清掉密码、验证码和错误：上一屏的错误文案留到下一屏会指向错误的原因。 */
  const switchMode = (m: Mode) => {
    setMode(m);
    setStep('credentials');
    setPassword('');
    setCode('');
    setError('');
    setNotice('');
    setCooldown(0);
  };

  const backToCredentials = () => {
    setStep('credentials');
    setCode('');
    setError('');
    setCooldown(0);
  };

  const register = mode === 'register';
  const onCode = step === 'code';
  const disabled = loading
    || !email.trim()
    || (onCode ? code.length !== CODE_LEN : !password)
    || (register && !onCode && password.length < MIN_PASSWORD_LEN);

  const primaryLabel = loading
    ? '请稍候...'
    : onCode
      ? (register ? '注册并进入' : '重设密码')
      : mode === 'login' ? '登录' : '发送验证码';

  return (
    <div className="login-page">
      <div className="login-card" style={{ width: 420 }}>
        <h1>🧑 账号</h1>
        <p className="login-subtitle">{SUBTITLE[mode]}</p>

        {!onCode && (
          <>
            <input
              className="login-input"
              style={{ letterSpacing: 0 }}
              type="email"
              placeholder="邮箱"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
              autoFocus
              disabled={loading}
            />

            {register && (
              <input
                className="login-input"
                style={{ letterSpacing: 0, marginTop: 10 }}
                placeholder="昵称（留空则用邮箱前缀）"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
                disabled={loading}
              />
            )}

            <input
              className="login-input"
              style={{ letterSpacing: 0, marginTop: 10 }}
              type="password"
              placeholder={
                mode === 'reset'
                  ? `新密码（至少 ${MIN_PASSWORD_LEN} 位）`
                  : register ? `密码（至少 ${MIN_PASSWORD_LEN} 位）` : '密码'
              }
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
              maxLength={128}
              disabled={loading}
            />
          </>
        )}

        {onCode && (
          <>
            {/* 复用 .login-input 的居中与字距是**刻意**的 —— 这是它唯一正确的
                场合（其余调用方都在写内联 letterSpacing:0 去对抗它，因为它是
                给老的访问码输入框设计的）。.login-code 只调字号与字距。 */}
            <input
              className="login-input login-code"
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={CODE_LEN}
              placeholder={'0'.repeat(CODE_LEN)}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, CODE_LEN))}
              onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
              autoFocus
              disabled={loading}
            />
            <p style={{ fontSize: '0.78rem', color: '#64748b', margin: '10px 0 0' }}>
              {/* reset 用「若已注册」而不是直接断言：后端对未注册邮箱也返回同样的
                  200 且不发信（防止拿这个接口枚举已注册邮箱），所以前端无从知道
                  信到底发没发。这里不能替服务端吹一个它保证不了的牛。 */}
              {mode === 'reset'
                ? `若 ${sentTo} 已注册，验证码已发出，10 分钟内有效。没收到请检查垃圾邮件。`
                : `验证码已发送至 ${sentTo}，10 分钟内有效。没收到请检查垃圾邮件。`}
            </p>
            <button
              type="button"
              className="login-btn"
              style={{ marginTop: 6, background: '#94a3b8' }}
              onClick={requestCode}
              disabled={loading || cooldown > 0}
            >
              {cooldown > 0 ? `${cooldown}s 后可重新发送` : '重新发送验证码'}
            </button>
          </>
        )}

        <p style={{ fontSize: '0.78rem', color: '#64748b', margin: '10px 0 0' }}>
          {mode === 'reset'
            ? '重设密码不会让已登录的设备下线，请同时在其他设备上退出登录。'
            : '注册需要邮箱验证码。忘记密码可以用邮箱验证码重设。'}
        </p>

        {notice && (
          <div className="login-error" style={{ color: '#0f766e' }}>{notice}</div>
        )}
        {error && <div className="login-error">{error}</div>}

        <button className="login-btn" onClick={submit} disabled={disabled}>
          {primaryLabel}
        </button>

        {onCode ? (
          <button
            className="login-btn"
            style={{ marginTop: 6, background: '#94a3b8' }}
            onClick={backToCredentials}
            disabled={loading}
          >
            返回修改
          </button>
        ) : (
          <>
            <button
              className="login-btn"
              style={{ marginTop: 6, background: '#94a3b8' }}
              onClick={() => switchMode(mode === 'login' ? 'register' : 'login')}
              disabled={loading}
            >
              {mode === 'login' ? '没有账号？去注册' : '已有账号？去登录'}
            </button>
            {mode === 'login' && (
              <button
                type="button"
                className="login-btn"
                style={{ marginTop: 6, background: '#94a3b8' }}
                onClick={() => switchMode('reset')}
                disabled={loading}
              >
                忘记密码？
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
