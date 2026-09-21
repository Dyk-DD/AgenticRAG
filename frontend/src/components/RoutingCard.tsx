import { useId, useState } from 'react';
import type { RoutingInfo } from '../types';
import { percent, strategyLabel } from '../lib/routing';

/**
 * 「推理过程」折叠面板，取代原先的 .routing-card。
 *
 * 折叠改由 <button> 承载，而不是原来的「整张卡 div 的 onClick」：后者键盘不可达，
 * 也没有 aria-expanded。点击热区由整卡收缩到按钮是有意为之。
 *
 * 面板体始终渲染、用 hidden 切换，而不是条件渲染 —— 这样 aria-controls 指向的
 * id 在折叠时也真实存在，不会指向一个不存在的节点。
 */
export default function RoutingCard({ routing }: { routing: RoutingInfo }) {
  const [expanded, setExpanded] = useState(false);
  const bodyId = useId();

  return (
    <div className="reasoning">
      <button
        type="button"
        className="reasoning-toggle"
        aria-expanded={expanded}
        aria-controls={bodyId}
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="reasoning-chevron" aria-hidden="true">
          {expanded ? '▾' : '▸'}
        </span>
        <span className="reasoning-label">推理过程</span>
        <span className="reasoning-hint">{strategyLabel(routing.strategy)}</span>
      </button>

      <div className="reasoning-body" id={bodyId} hidden={!expanded}>
        <p className="reasoning-text">{routing.reasoning}</p>
        <div className="reasoning-metrics">
          <span>
            复杂度 <b>{percent(routing.complexity)}</b>
          </span>
          <span>
            关系密集度 <b>{percent(routing.intensity)}</b>
          </span>
          <span>
            调度策略 <b>{strategyLabel(routing.strategy)}</b>
          </span>
        </div>
      </div>
    </div>
  );
}
