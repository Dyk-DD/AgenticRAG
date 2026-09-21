import { create } from 'zustand';
import type { Message, RoutingInfo, Session } from '../types';
import * as api from '../api/client';

interface ChatState {
  messages: Message[];
  sessions: Session[];
  sessionId: string | null;
  loading: boolean;
  streaming: boolean;
  error: string | null;
  routing: RoutingInfo | null;

  addMessage: (msg: Message) => void;
  appendToLast: (text: string) => void;
  setLoading: (v: boolean) => void;
  setStreaming: (v: boolean) => void;
  setError: (e: string | null) => void;
  setRouting: (r: RoutingInfo | null) => void;
  clearMessages: () => void;

  loadSessions: () => Promise<void>;
  newSession: () => Promise<void>;
  removeSession: (id: string) => Promise<void>;
  renameSession: (id: string, title: string) => Promise<void>;
  pinSession: (id: string, pinned: boolean) => Promise<void>;
  activateSession: (id: string) => Promise<void>;
  sendMessage: (question: string) => Promise<void>;
  stopStreaming: () => void;
}

let abortController: AbortController | null = null;

let msgIdCounter = 0;
function nextId() {
  msgIdCounter += 1;
  return `msg_${Date.now()}_${msgIdCounter}`;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  sessions: [],
  sessionId: null,
  loading: false,
  streaming: false,
  error: null,
  routing: null,

  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  appendToLast: (text) =>
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, content: last.content + text };
      }
      return { messages: msgs };
    }),
  setLoading: (v) => set({ loading: v }),
  setStreaming: (v) => set({ streaming: v }),
  setError: (e) => set({ error: e }),
  setRouting: (r) => set({ routing: r }),
  clearMessages: () => set({ messages: [] }),

  loadSessions: async () => {
    try {
      const data = await api.fetchSessions();
      set({ sessions: data.sessions || [] });
    } catch {
      // ignore
    }
  },

  activateSession: async (sessionId: string) => {
    try {
      await api.activateSession(sessionId);
      const detail = await api.fetchSessionDetail(sessionId);
      // 后端 TurnRecord 的字段是 timestamp（不是 created_at）—— 读错字段会
      // 让 Date.parse(undefined) 恒为 NaN，每条恢复的消息时间戳都回落成「现在」
      const messages: Message[] = (detail.turns || []).flatMap((t: any) => [
        { id: nextId(), role: 'user' as const, content: t.question, timestamp: Date.parse(t.timestamp) || Date.now() },
        { id: nextId(), role: 'assistant' as const, content: t.answer, timestamp: Date.now() },
      ]);
      set({ sessionId, messages, loading: false, streaming: false, error: null, routing: null });
    } catch {
      set({ error: '无法加载会话' });
    }
  },

  newSession: async () => {
    try {
      const data = await api.createSession();
      set({ sessionId: data.session_id, messages: [], routing: null, error: null });
      await get().loadSessions();
    } catch {
      // ignore
    }
  },

  removeSession: async (id) => {
    try {
      await api.deleteSession(id);
      await get().loadSessions();
    } catch {
      // ignore
    }
  },

  renameSession: async (id, title) => {
    const prevTitle = get().sessions.find((s) => s.id === id)?.title;
    const patch = (t: string) =>
      set((s) => ({ sessions: s.sessions.map((x) => (x.id === id ? { ...x, title: t } : x)) }));

    // 乐观更新：先本地改，成功后采纳服务端清洗/截断过的标题，失败回滚原值。
    // 不用 await 完再 loadSessions() —— 那是整表重取，会按 created_at DESC 重排
    // 列表，改名时看起来像界面在抖；而且重排会把改名的会话挪位置。
    patch(title);
    try {
      const data = await api.renameSession(id, title);
      if (typeof data?.title === 'string') patch(data.title);
    } catch (e) {
      if (prevTitle !== undefined) patch(prevTitle);
      set({ error: e instanceof Error ? e.message : '改名失败' });
    }
  },

  pinSession: async (id, pinned) => {
    try {
      await api.pinSession(id, pinned);
      // ⚠️ 这里**必须**整表重取，不能照抄 renameSession 的乐观 patch。
      // renameSession 刻意不重取，理由是「整表重取会按 created_at 重排，看起来像
      // 界面在抖」—— 而置顶的重排**就是功能本身**。乐观 patch 只能改 pinned_at
      // 字段、改不动顺序，结果是「置顶了但没动」，用户会以为没生效再点一次。
      await get().loadSessions();
    } catch (e) {
      set({ error: e instanceof Error ? e.message : '置顶失败' });
    }
  },

  sendMessage: async (question: string) => {
    const state = get();
    if (state.loading || state.streaming) return;

    const userMsg: Message = {
      id: nextId(),
      role: 'user',
      content: question,
      timestamp: Date.now(),
    };
    set({ messages: [...state.messages, userMsg], loading: true, streaming: true, error: null, routing: null });

    const assistantMsg: Message = {
      id: nextId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };
    set((s) => ({ messages: [...s.messages, assistantMsg] }));

    abortController = api.chatStream(
      question,
      (routingData) => {
        const routing: RoutingInfo = {
          strategy: routingData.strategy,
          complexity: routingData.complexity,
          intensity: routingData.intensity,
          reasoning: routingData.reasoning,
        };
        set((s) => {
          const msgs = [...s.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === 'assistant') {
            msgs[msgs.length - 1] = { ...last, routing };
          }
          return { messages: msgs, routing };
        });
      },
      (text) => {
        get().appendToLast(text);
      },
      (doneData) => {
        set({ loading: false, streaming: false, sessionId: doneData.session_id || state.sessionId });
        // 首条提问在后端顺带定下了会话标题，重取一次列表让侧栏显示出来 ——
        // 新建的会话此前根本不在列表里（后端是在第一次提问时才建会话的）。
        // 这里重排可以接受：列表内容确实变了，不是改名那种纯展示抖动。
        get().loadSessions();
      },
      (errMsg) => {
        set({ error: errMsg, loading: false, streaming: false });
      },
    );
  },

  stopStreaming: () => {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    set({ loading: false, streaming: false });
  },
}));
