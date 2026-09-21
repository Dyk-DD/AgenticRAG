import { useState } from 'react';
import { getToken, setToken } from '../api/config';

interface Props {
  onLogin: () => void;
}

export default function LoginPage({ onLogin }: Props) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!password.trim()) return;
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/auth', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
        },
        body: JSON.stringify({ password: password.trim() }),
      });

      if (res.ok) {
        const data = await res.json();
        setToken(data.token);
        onLogin();
      } else {
        // 解析失败本身就说明问题不在密码上：nginx / 隧道返回的 502、504 是 HTML，
        // 不是 JSON。以前这里 catch 成 {} 再取 detail 为空，就把「后端没起来」
        // 报成了「密码错误」——用户会去反复改密码。catch 分支那句更中肯的话也
        // 永远走不到，因为 fetch 拿到 502 是 resolve 而不是 reject。
        const data = await res.json().catch(() => null);
        setError(
          data
            ? data.detail || '密码错误，请重试'
            : `后端返回异常（HTTP ${res.status}），请确认服务已启动`,
        );
      }
    } catch {
      setError('无法连接到后端，请检查后端服务是否启动');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <h1>🏥 智能问答系统</h1>
        <p className="login-subtitle">请输入访问密码</p>

        <input
          type="password"
          className="login-input"
          placeholder="访问密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleLogin(); }}
          autoFocus
          disabled={loading}
        />

        {error && <div className="login-error">{error}</div>}

        <button
          className="login-btn"
          onClick={handleLogin}
          disabled={loading || !password.trim()}
        >
          {loading ? '验证中...' : '进入系统'}
        </button>
      </div>
    </div>
  );
}
