export interface RoutingInfo {
  strategy: string;
  complexity: number;
  intensity: number;
  reasoning: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  routing?: RoutingInfo;
  timestamp: number;
}

export interface Session {
  id: string;
  /** 首条提问（或用户改过的名字）。后端保证是字符串，可能为空串。 */
  title: string;
  start_time: string;
  end_time: string;
  turn_count: number;
  /** 置顶时刻（YYYY-MM-DDTHH:MM:SS）；空串 = 未置顶。 */
  pinned_at: string;
}

export interface SessionDetail {
  id: string;
  title: string;
  start_time: string;
  end_time: string;
  turns: TurnRecord[];
}

export interface TurnRecord {
  /**
   * 后端 qa_records.turn_index 原值。分享功能按它挑轮次 —— 不能用数组下标，
   * 下标是隐式契约，一次前端过滤就会让它静默错位。
   * **不保证连续**：并发下可能出现两条 turn_index = 0，所以只能按值相等匹配，
   * 不能拿它当序号或数组索引。
   */
  turn_index: number;
  question: string;
  answer: string;
  strategy: string;
  complexity: number;
  timestamp: string;
}

/**
 * 分享页能拿到的字段。**这是后端白名单的结果**（qa_database.SHARE_TURN_FIELDS）：
 * 刻意不含 retrieved_docs 与 routing_reasoning —— 前者是知识库内部标识与检索
 * 分数，后者是内部推理链，都不该出现在一条完全公开的链接里。
 * 这里不声明它们，是为了让「想渲染就得先改后端白名单」成为一个编译期障碍。
 */
export interface ShareTurn {
  turn_index: number;
  question: string;
  answer: string;
  strategy: string;
  complexity: number;
  timestamp: string;
}

export interface ShareDetail {
  title: string;
  created_at: string;
  turns: ShareTurn[];
}

export interface Stats {
  qa_pairs: number;
  departments: number;
  milvus_rows: number;
  total_queries: number;
  route_distribution?: {
    traditional: number;
    graph_rag: number;
    combined: number;
  };
}
