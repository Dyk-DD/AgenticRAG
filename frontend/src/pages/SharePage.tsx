import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import * as api from '../api/client';
import type { ShareDetail } from '../types';
import { percent, strategyLabel } from '../lib/routing';

/**
 * 公开分享页（/s/:token）。**不经过 ChatGate** —— 拿到链接的人没有访问密码也
 * 没有账号，这正是「完全公开」这个决定的含义。
 *
 * 三种状态：加载中 / 有效 / 失效。失效不与「网络出错」混为一谈：前者是这条链接
 * 真的没了（已撤销或会话被删），后者是暂时打不开，把两者都写成「已失效」会让
 * 用户去问分享者要新链接，而其实刷新一下就好。
 */
export default function SharePage() {
  const { token = '' } = useParams();
  const [data, setData] = useState<ShareDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [gone, setGone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const d = await api.fetchShare(token);
        if (alive) setData(d);
      } catch (e) {
        if (!alive) return;
        // 404 是「不存在 / 已撤销 / 行损坏」，后端刻意不区分原因 ——
        // 区分了就等于给「这个 token 曾经存在过」提供一条探测通道。
        if (e instanceof api.ApiError && e.status === 404) setGone(true);
        else setError(e instanceof Error ? e.message : '加载失败');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [token]);

  return (
    <div className="share-page">
      <header className="share-header">
        <h1 className="share-header-title">{data?.title || '分享的对话'}</h1>
        {data && (
          <p className="share-header-meta">
            共 {data.turns.length} 轮 · 分享于 {new Date(data.created_at).toLocaleString()}
          </p>
        )}
      </header>

      {loading && <p className="empty-hint">加载中...</p>}

      {!loading && gone && (
        <div className="share-empty">
          <p className="share-empty-title">该分享已被取消或失效</p>
          <p className="share-empty-sub">分享者可能撤销了链接，或删除了对应的会话。</p>
        </div>
      )}

      {!loading && error && (
        <div className="share-empty">
          <p className="share-empty-title">暂时无法打开</p>
          <p className="share-empty-sub">{error}</p>
        </div>
      )}

      {!loading && data && (
        <>
          <div className="turn-list">
            {data.turns.map((t) => (
              // key 用 turn_index：它不保证连续（并发下会有两条 = 0），但同一份
              // 快照里不会重复，且比数组下标稳定 —— 快照存的是固定内容，没有
              // 增删，但用业务键仍然更贴合语义。
              <div key={t.turn_index} className="turn-item">
                <div className="turn-question">
                  <strong>Q:</strong> {t.question}
                </div>
                <div className="turn-answer">
                  <strong>A:</strong> {t.answer}
                </div>
                <div className="turn-meta">
                  策略: {strategyLabel(t.strategy)} | 复杂度: {percent(t.complexity)} | {t.timestamp}
                </div>
              </div>
            ))}
          </div>

          <footer className="share-footer">
            {/* 这句不是装饰：分享页脱离了应用上下文，访问者可能直接落到一段医学
                回答上而毫无框架 —— 与公开主页页脚同一句。 */}
            <p className="share-disclaimer">
              本系统输出由模型生成，仅供研究与技术演示参考，不构成医疗建议。
            </p>
            <div className="share-footer-links">
              <Link to="/">返回主页</Link>
              <Link to="/chat">进入智能问答系统</Link>
            </div>
          </footer>
        </>
      )}
    </div>
  );
}
