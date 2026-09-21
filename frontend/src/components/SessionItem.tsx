import { useEffect, useId, useRef, useState } from 'react';
// 起个别名：下面监听器的 e 是 DOM 的 MouseEvent，两者同名会打架
import type { MouseEvent as ReactMouseEvent } from 'react';
import type { Session } from '../types';

/** 与后端 qa_database.TITLE_MAX_LEN 一致，本地先挡一道。 */
const TITLE_MAX_LEN = 40;

interface Props {
  session: Session;
  active: boolean;
  onActivate: () => void;
  onRename: (title: string) => void;
  onPin: (pinned: boolean) => void;
  onShare: () => void;
  onDelete: () => void;
}

/**
 * 侧栏里的单条会话：标题 + 轮次数 + 「⋯」菜单（置顶 / 重命名 / 分享 / 删除）。
 *
 * 本组件把原先的 RenameableTitle 折了进来。那个组件的存在理由是「侧栏与历史
 * 记录页共用」，历史页删掉后它只剩一个消费者，同时「重命名」从行内铅笔移进
 * 菜单后编辑态必须能被菜单触发 —— 两个理由都指向合并。
 */
export default function SessionItem({
  session,
  active,
  onActivate,
  onRename,
  onPin,
  onShare,
  onDelete,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  /** 菜单相对 .sidebar（它的包含块）的 top 偏移，px。 */
  const [menuTop, setMenuTop] = useState(0);
  const [openUp, setOpenUp] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  // 触发按钮与浮层同在这个 ref 里：它既是「点击外部」的判定范围，也是下面
  // 测量锚点的来源。理由与 .settings-menu 那处完全一致 —— 判定范围若不含
  // 触发按钮，按在按钮上的那一下会被外部监听先关掉、再被 onClick 打开。
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  const title = session.title || '新会话';
  const pinned = Boolean(session.pinned_at);

  /**
   * 打开菜单前先测量行在侧栏里的位置。
   *
   * 为什么非测不可：菜单的包含块被**刻意**放在 .sidebar（见 index.css 里
   * .sidebar 的 position: relative），而不是 .session-item —— 否则包含块会落进
   * 滚动容器 .session-list 内部，菜单被裁成两半。包含块挪出滚动容器之后，
   * top 就得以侧栏为原点了，这个换算是免不了的，不是因为偷懒。
   */
  const toggleMenu = () => {
    if (menuOpen) {
      setMenuOpen(false);
      return;
    }
    const row = rowRef.current;
    const sidebar = row?.closest('.sidebar');
    if (!row || !sidebar) return;
    const r = row.getBoundingClientRect();
    const s = sidebar.getBoundingClientRect();
    // 行落在侧栏下半区就向上弹：菜单有四项、约 150px 高，贴着底部的行若向下
    // 弹会伸到视口外面，「删除」就点不到了。用 translateY(-100%) 翻转，不必
    // 知道菜单实际高度。
    const up = r.top + r.height / 2 > s.top + s.height / 2;
    setOpenUp(up);
    setMenuTop(up ? r.top - s.top - 4 : r.bottom - s.top + 4);
    setMenuOpen(true);
  };

  // 关闭时机。setState 只出现在 DOM 事件的回调里，不写在 effect 本体 ——
  // 本仓库 react-hooks/set-state-in-effect 是 error。
  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    const onMouseDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) close();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    // 滚动必须监听，而且用 capture：scroll 不冒泡，capture 才能收到 .session-list
    // 自己的滚动。不监听的话行会滚走而菜单留在原地，而菜单项里有「分享 / 删除」
    // 这类破坏性操作 —— 用户看着 A 行的菜单点删除，实际作用在 B 行。
    // resize 同理：重排之后测量值就失效了。
    document.addEventListener('scroll', close, { capture: true });
    window.addEventListener('resize', close);
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('scroll', close, { capture: true });
      window.removeEventListener('resize', close);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [menuOpen]);

  const commit = () => {
    const next = draft.trim();
    setEditing(false);
    // 空值或没改动就不发请求：后端会 400，而用户只是想退出编辑
    if (next && next !== session.title) onRename(next);
  };

  const startRename = () => {
    setDraft(session.title);
    setEditing(true);
    setMenuOpen(false);
  };

  // 菜单项的公共部分：每一项都必须 stopPropagation，否则会连带触发整行的
  // 「切换会话」—— 那意味着点「删除」会先切到那条会话再删掉它。
  const act = (fn: () => void) => (e: ReactMouseEvent) => {
    e.stopPropagation();
    setMenuOpen(false);
    fn();
  };

  return (
    <div
      ref={rowRef}
      className={`session-item ${active ? 'active' : ''}`}
      onClick={() => {
        // 编辑态下不切换会话：点空白处让输入框卸载，用户刚敲的字就没了
        if (editing) return;
        onActivate();
      }}
    >
      <span className="session-info">
        {editing ? (
          <input
            className="title-input"
            value={draft}
            autoFocus
            maxLength={TITLE_MAX_LEN}
            onChange={(e) => setDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') commit();
              else if (e.key === 'Escape') setEditing(false);
            }}
          />
        ) : (
          // 侧栏很窄而标题最长 40 字，必溢出，所以省略号 + title 属性给全文
          <span className="session-title" title={title}>
            {pinned && <span className="session-pin-mark" aria-label="已置顶">📌</span>}
            {title}
          </span>
        )}
        <span className="session-turns">{session.turn_count} 轮</span>
      </span>

      {/* position 必须是 static（默认）：给这层或 .session-item 加 relative
          会把菜单的包含块拽回滚动容器内部，菜单当场被裁。见 index.css 的注释。 */}
      <div className="session-menu-anchor" ref={menuRef}>
        <button
          type="button"
          className="btn-session-menu"
          aria-label="会话操作"
          aria-haspopup="true"
          aria-expanded={menuOpen}
          aria-controls={menuOpen ? menuId : undefined}
          onClick={(e) => {
            e.stopPropagation();
            toggleMenu();
          }}
        >
          ⋯
        </button>

        {/* 不写 role="menu"：ARIA menu 带 roving tabindex + 方向键的契约，
            只写 role 不实现键盘导航比不写更糟。这里是 disclosure 模式。 */}
        {menuOpen && (
          <div
            className={`session-menu ${openUp ? 'session-menu-up' : ''}`}
            id={menuId}
            style={{ top: menuTop }}
          >
            <button type="button" className="session-menu-item" onClick={act(() => onPin(!pinned))}>
              <span aria-hidden="true">📌</span>
              {pinned ? '取消置顶' : '置顶'}
            </button>
            <button type="button" className="session-menu-item" onClick={act(startRename)}>
              <span aria-hidden="true">✎</span>重命名
            </button>
            <button
              type="button"
              className="session-menu-item"
              onClick={act(onShare)}
              // 还没提问的会话没有可分享的轮次，后端也会 400。禁用比报错清楚。
              disabled={session.turn_count === 0}
            >
              <span aria-hidden="true">🔗</span>分享
            </button>
            <button
              type="button"
              className="session-menu-item danger"
              onClick={act(onDelete)}
            >
              <span aria-hidden="true">🗑</span>删除
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
