import { useEffect, useId, useState } from 'react';
import * as api from '../api/client';
import type { SessionDetail } from '../types';

interface Props {
  sessionId: string;
  onClose: () => void;
}

/**
 * 分享对话框：列出会话里每一轮，勾选若干轮生成一条公开链接。
 *
 * ⚠️ 本组件**绝不能**挂在 .sidebar 内部（ChatLayout 把它渲染在 .main-content 里）。
 * 侧栏只有 280px，装不下一份问答列表，所以它必然是 inset:0 的全屏遮罩；而窄屏
 * 抽屉态的 .sidebar 带 transform，**会成为一个 fixed 后代的包含块** —— 全屏浮层
 * 会被压进 280px 的抽屉里，看起来像是「弹窗坏了」。
 */
export default function ShareDialog({ sessionId, onClose }: Props) {
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** 生成成功后的结果；非空即进入「已生成」视图。 */
  const [created, setCreated] = useState<{ url: string; token: string; count: number } | null>(null);
  const [copied, setCopied] = useState(false);
  const titleId = useId();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await api.fetchSessionDetail(sessionId);
        if (!alive) return;
        setDetail(data);
        // 默认全选：分享整段对话是最常见的意图，取消比逐个勾省事
        setSelected(new Set((data.turns || []).map((t) => t.turn_index)));
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : '加载会话失败');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [sessionId]);

  // Esc 关闭。点遮罩关闭由遮罩自己的 onClick 承担（见下方 JSX）。
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const toggle = (turnIndex: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(turnIndex)) next.delete(turnIndex);
      else next.add(turnIndex);
      return next;
    });
  };

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      // 传真实 turn_index，不是数组下标（见 api/client.ts 的注释）
      const data = await api.createShare(sessionId, [...selected]);
      setCreated({
        // 用后端给的 path 而不是前端自己拼路由前缀
        url: `${window.location.origin}${data.path}`,
        token: data.token,
        count: data.turn_count,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : '生成链接失败');
    }
    setBusy(false);
  };

  const copy = async () => {
    if (!created) return;
    try {
      await navigator.clipboard.writeText(created.url);
      setCopied(true);
    } catch {
      // 非安全上下文 / 未授权剪贴板：不报错，链接就在输入框里，手动复制即可
      setCopied(false);
    }
  };

  const revoke = async () => {
    if (!created) return;
    setBusy(true);
    try {
      await api.revokeShare(created.token);
      // 撤销后回勾选视图，用户可以改选轮次重新生成
      setCreated(null);
      setCopied(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : '撤销失败');
    }
    setBusy(false);
  };

  const turns = detail?.turns || [];

  return (
    <div className="share-dialog-scrim" onClick={onClose}>
      {/* stopPropagation：点面板内部不该关掉自己 */}
      <div
        className="share-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="share-dialog-head">
          <h2 id={titleId} className="share-dialog-title">
            🔗 分享会话
          </h2>
          <button type="button" className="share-dialog-close" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </div>

        {loading && <p className="empty-hint">加载中...</p>}

        {!loading && !created && (
          <>
            <p className="share-dialog-hint">
              选择要分享的对话。链接是<b>公开</b>的 —— 任何拿到它的人都能查看，
              无需登录。
            </p>
            <div className="share-turn-list">
              {turns.map((t) => (
                <label key={t.turn_index} className="share-turn">
                  <input
                    type="checkbox"
                    checked={selected.has(t.turn_index)}
                    onChange={() => toggle(t.turn_index)}
                  />
                  <span className="share-turn-text">
                    <span className="share-turn-q">{t.question}</span>
                    <span className="share-turn-a">{t.answer}</span>
                  </span>
                </label>
              ))}
              {turns.length === 0 && <p className="empty-hint">这个会话还没有对话</p>}
            </div>
          </>
        )}

        {created && (
          <div className="share-result">
            <p className="share-dialog-hint">
              链接已生成（{created.count} 轮对话）。把它发给任何人都可以打开。
            </p>
            <input className="share-url" readOnly value={created.url} onFocus={(e) => e.target.select()} />
          </div>
        )}

        {error && <p className="share-dialog-error">⚠️ {error}</p>}

        <div className="share-dialog-actions">
          {!created ? (
            <>
              <button type="button" className="btn-secondary" onClick={onClose}>
                取消
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={generate}
                disabled={busy || selected.size === 0}
              >
                {busy ? '生成中...' : `生成链接（${selected.size} 轮）`}
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn-secondary" onClick={revoke} disabled={busy}>
                撤销链接
              </button>
              <button type="button" className="btn-primary" onClick={copy}>
                {copied ? '已复制' : '复制链接'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
