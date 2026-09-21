/**
 * routing 字段的展示映射与格式化。
 *
 * 集中成一个模块，是因为原先同一个 strategy 有三条彼此不一致的呈现路径：
 * ChatMessage 里一份中英映射、RoutingCard 里一份 emoji 映射、HistoryPage 直接
 * 把英文原样打到界面上。改一处必漏另外两处。
 * （HistoryPage 已删，但这条约束没变 —— 现存的两个消费者是 RoutingCard 与
 * 公开分享页 SharePage，仍然是两个界面。）
 *
 * 独立成 .ts 而不是塞进组件文件：本仓库 react-refresh/only-export-components
 * 是 error，组件文件里导出非组件会让该文件的 Fast Refresh 整体失效。
 */

export const STRATEGY_LABELS: Record<string, string> = {
  hybrid_traditional: '基础医学检索',
  graph_rag: '复杂临床路径推理',
  combined: '图谱+向量综合研判',
};

/** 未知策略回落到原始值 —— 后端将来加了新策略，界面显示英文也比显示 undefined 强。 */
export function strategyLabel(strategy: string): string {
  return STRATEGY_LABELS[strategy] || strategy;
}

/** 0.42 -> "42%"。complexity / intensity 都要走它，避免各处重复写 toFixed。 */
export function percent(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}
