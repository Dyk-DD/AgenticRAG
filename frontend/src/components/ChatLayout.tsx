import { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react';
import { Link, useMatch, Outlet } from 'react-router-dom';
import { useChatStore } from '../store/chatStore';
import { clearToken, getPatientId, getPatientLabel, clearPatientId } from '../api/config';
import SessionItem from './SessionItem';
import ShareDialog from './ShareDialog';

/** 侧栏折叠偏好。窄屏抽屉不写这个键 —— 它是宽屏专属的持久化偏好。 */
const SIDEBAR_PREF_KEY = 'agentic_rag_sidebar_collapsed';

function readCollapsedPref(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_PREF_KEY) === '1';
  } catch {
    // 隐私模式 / 禁用了站点数据时 localStorage 会直接抛，不能让它挡住首屏
    return false;
  }
}

function writeCollapsedPref(collapsed: boolean): void {
  try {
    localStorage.setItem(SIDEBAR_PREF_KEY, collapsed ? '1' : '0');
  } catch {
    // 同上：存不下就算了，本次会话内仍然照常折叠
  }
}

const NARROW_QUERY = '(max-width: 900px)';
let narrowMql: MediaQueryList | null = null;
function getNarrowMql(): MediaQueryList {
  if (!narrowMql) narrowMql = window.matchMedia(NARROW_QUERY);
  return narrowMql;
}

// subscribe / getSnapshot 都定义在模块顶层，引用天然稳定（useSyncExternalStore
// 每次渲染拿到新函数就会重新订阅）。
function subscribeNarrow(cb: () => void) {
  const mql = getNarrowMql();
  mql.addEventListener('change', cb);
  return () => mql.removeEventListener('change', cb);
}
// 必须返回**原始布尔值**：返回 mql 对象会让 React 认为快照一直在变，无限重渲染。
function getNarrowSnapshot(): boolean {
  return getNarrowMql().matches;
}

/**
 * 窄屏判定（≤900px，与 index.css 里那个断点同一个值）。
 *
 * 用 useSyncExternalStore + matchMedia，而不是 effect 里挂 resize 监听 —— 后者
 * 必须在 effect 本体里 setState，而本仓库 react-hooks/set-state-in-effect 是
 * error。这里刻意不导出：只有本组件用得到，导出会触发
 * react-refresh/only-export-components。
 */
function useIsNarrow(): boolean {
  return useSyncExternalStore(subscribeNarrow, getNarrowSnapshot);
}

export default function ChatLayout() {
  // 用 useMatch 而不是 location.pathname === '/chat' 字面比较：浏览器地址栏
  // 停在 /chat/（尾斜杠）时，路由照样匹配，字面比较却会失败 —— 那样会话列表
  // 会静默消失，用户看到的是「我的历史记录没了」，且不抛任何错。
  // 历史页删掉后 ChatLayout 只剩 /chat 一个挂载点，这个守卫今天恒为真；保留它
  // 是因为它表达的「会话列表属于对话视图」这条约束仍然成立。
  const onChat = useMatch('/chat') !== null;
  const {
    sessions, sessionId, loadSessions, newSession, removeSession,
    renameSession, pinSession, activateSession,
  } = useChatStore();

  // 正在分享的会话 id。**状态提在这里而不是 SessionItem 里**：ShareDialog 必须
  // 渲染在 .main-content 内（见 ShareDialog 顶部的说明），不能挂在侧栏里，
  // 所以触发它的行只能把意图往上报。
  const [shareId, setShareId] = useState<string | null>(null);

  const [menuOpen, setMenuOpen] = useState(false);
  // 触发按钮和浮层包在同一个 ref 里：它既是「点击外部」的判定范围（见下方
  // mousedown 的注释），也是浮层绝对定位的锚点。
  const menuRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const sidebarId = useId();

  // 惰性初始化读 localStorage，不写成 effect 里 setState
  // （react-hooks/set-state-in-effect 在本仓库是 error）。
  const [prefCollapsed, setPrefCollapsed] = useState(readCollapsedPref);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const isNarrow = useIsNarrow();

  // 一个布尔量表达不了「宽屏=收成 0 宽」与「窄屏=抽屉默认关闭」两种语义，
  // 混用会让移动端首屏就弹出抽屉。合成后交给 CSS 的只有 .sidebar-collapsed
  // 这一个类，两个按钮的可见性因此不必知道自己是宽是窄（见 index.css）。
  const collapsed = isNarrow ? !drawerOpen : prefCollapsed;

  const toggleSidebar = () => {
    if (isNarrow) {
      // 抽屉开关不入库：它是窄屏下的临时态，与宽屏偏好不是一回事
      setDrawerOpen((v) => !v);
      return;
    }
    const next = !prefCollapsed;
    setPrefCollapsed(next);
    writeCollapsedPref(next);
    // 收起时顺手关掉设置浮层：侧栏会变 inert，浮层留在里面既够不着、也会被
    // overflow: hidden 裁掉
    if (next) setMenuOpen(false);
  };

  // 优先显示名字/邮箱；老身份没有 label，退回 id 前 12 位
  const patientLabel = getPatientLabel();
  const patientId = getPatientId();
  const accountLabel = patientLabel || `${patientId.slice(0, 12)}...`;

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // 点击外部关闭 + Esc 关闭。这是 src/ 里第一个浮层，四点写清楚：
  //   - 监听 mousedown 而不是 click：click 要等 mouseup，若按下后拖到别处松手，
  //     事件目标会落在按下点之外，一次拖拽就把浮层「莫名」关掉了。
  //   - 判定范围用 menuRef.contains 而不是「打开时挂上 document 监听」：触发按钮
  //     也在 menuRef 里，所以按在按钮上的那一下不会被外部监听先关掉、再被 onClick
  //     打开（那会表现为「点按钮关不掉」）。按钮的开与关只由 onClick 决定。
  //   - setState 只出现在 DOM 事件的回调里，不写在 effect 本体 —— 本仓库
  //     react-hooks/set-state-in-effect 是 error。
  //   - 监听器成对移除：本组件卸载后不能还挂在 document 上。
  useEffect(() => {
    if (!menuOpen) return;
    const onMouseDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    // Esc 关闭并把焦点还给触发按钮，否则焦点会掉到 body 上，键盘用户失去落点
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      setMenuOpen(false);
      toggleRef.current?.focus();
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [menuOpen]);

  return (
    <div className={`app-layout ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {/* inert 不是可选的：折叠后侧栏里全是仍可聚焦的控件（新会话、会话项、
          设置按钮），键盘用户按 Tab 会掉进一块 0 宽、看不见的区域。
          不能用 aria-hidden 代替 —— 它不移除焦点顺序。
          collapsed 为 false 时传 undefined 而不是 false，免得往 DOM 上写 inert="false"。 */}
      <aside className="sidebar" id={sidebarId} inert={collapsed || undefined}>
        <div className="sidebar-header">
          {/* 品牌行三段：返回主页 / 字标 / 收起。
              字标本身**不是**链接 —— 它在侧栏最显眼的位置，误点会直接离开应用，
              而「离开应用」是个有后果的动作，不该挂在一个看起来只是标题的元素上。
              返回主页改由左边那个明确的箭头承担（浮动按钮组里也有一个，折叠态下
              它是唯一的出口）。 */}
          <div className="sidebar-header-row">
            <Link to="/" className="sidebar-icon-btn" aria-label="返回主页" title="返回主页">
              ←
            </Link>
            <h2>🏥 智能问答系统</h2>
            <button
              type="button"
              className="sidebar-icon-btn"
              onClick={toggleSidebar}
              aria-label="收起侧边栏"
              aria-expanded={!collapsed}
              aria-controls={sidebarId}
            >
              «
            </button>
          </div>
          <button className="btn-new-session" onClick={newSession}>
            + 新会话
          </button>
        </div>

        {onChat && (
          <div className="session-list">
            <h3>会话列表</h3>
            {sessions.map((s) => (
              <SessionItem
                key={s.id}
                session={s}
                active={s.id === sessionId}
                onActivate={() => { activateSession(s.id); setDrawerOpen(false); }}
                onRename={(t) => renameSession(s.id, t)}
                onPin={(p) => pinSession(s.id, p)}
                onShare={() => setShareId(s.id)}
                // 不新增二次确认，与原先直接点「✕」的行为一致（见 README 的
                // 风险一节：移进菜单后误删代价变高，如果要加确认就加在这里，
                // 而不是在 store 里偷偷改成软删除）。
                onDelete={() => removeSession(s.id)}
              />
            ))}
            {sessions.length === 0 && <p className="empty-hint">暂无会话</p>}
          </div>
        )}

        <div className="sidebar-footer">
          {/* 浮层与触发按钮同在这个 div 里，且浮层相对**按钮**定位（bottom:100%），
              不是相对 .sidebar —— 侧边栏高 100vh，相对它定位会把浮层顶到视口外面去。 */}
          <div className="settings-menu" ref={menuRef}>
            <button
              type="button"
              ref={toggleRef}
              className="btn-settings"
              aria-expanded={menuOpen}
              aria-controls={menuOpen ? menuId : undefined}
              onClick={() => setMenuOpen((v) => !v)}
            >
              ⚙ 设置
              <span className="settings-caret">{menuOpen ? '▾' : '▴'}</span>
            </button>

            {/* 不写 role="menu"：ARIA menu 语义带 roving tabindex + 方向键 + Home/End
                的契约，只写 role 不实现键盘导航比不写更糟。这里是 disclosure 模式，
                aria-expanded 才是准确的表达。 */}
            {menuOpen && (
              <div className="settings-popover" id={menuId}>
                {/* 这里曾再挂一行裸 patient id（u_3f9a2b1c8d4e）。它重复了上面
                    已经在显示的信息（label 缺失时 accountLabel 本来就是 id 前缀），
                    而 label 存在时它又是一串没人会去核对的字符 —— 删掉。 */}
                <div className="settings-account">
                  <div className="settings-account-name" title={accountLabel}>
                    🧑 {accountLabel}
                  </div>
                </div>

                <div className="settings-divider" />

                {/* 切换账号只清患者身份、原地 reload：全局访问密码那道门保持打开，
                    AccountGate 会在当前 URL 上重新接住。必须用 reload 而不是路由
                    跳转 —— 它同时是令牌失效 / 身份被迁移后唯一的出口
                    （_get_patient 只验签不查库，这类令牌会一直「有效」并返回空列表，
                    用户看到的是「我的历史没了」）。 */}
                <button
                  type="button"
                  className="settings-item"
                  onClick={() => { clearPatientId(); window.location.reload(); }}
                >
                  <span className="settings-item-icon">🔄</span>
                  <span className="settings-item-text">
                    <span className="settings-item-label">切换账号</span>
                  </span>
                </button>

                {/* 退出比切换账号多清一个全局访问令牌（连访问密码那道门一起关掉），
                    且用整页导航回主页（而不是 reload）：既然现在有公开主页了，
                    「退出」该回到门户，而不是停在 /chat 的登录页。两项后果不同，
                    所以用红色把它单独标出来，避免误点。 */}
                <button
                  type="button"
                  className="settings-item danger"
                  onClick={() => { clearToken(); clearPatientId(); window.location.assign('/'); }}
                >
                  <span className="settings-item-icon">🚪</span>
                  <span className="settings-item-text">
                    <span className="settings-item-label">退出登录</span>
                  </span>
                </button>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* 窄屏抽屉的遮罩。decorative —— 关闭动作键盘上由 .sidebar-toggle 承担，
          所以对辅助技术隐藏，也不参与焦点顺序。 */}
      <div className="sidebar-scrim" aria-hidden="true" onClick={() => setDrawerOpen(false)} />

      <main className="main-content">
        {/* 浮动按钮组：侧栏收起 / 抽屉关闭时，这三个动作在侧栏里都够不着
            （侧栏此时是 inert 的），只能挪到主区左上角。
            ＋ 是后加的：折叠态原先无法新建会话 —— 唯一的入口「+ 新会话」
            在侧栏里，而侧栏正是被收起的那个。
            ← 同理：品牌字标已不可点，这里不补一个就在折叠态彻底没有回主页的路。 */}
        <div className="sidebar-toggle-group">
          <button
            type="button"
            className="sidebar-toggle"
            onClick={toggleSidebar}
            aria-label={collapsed ? '展开侧边栏' : '收起侧边栏'}
            aria-expanded={!collapsed}
            aria-controls={sidebarId}
          >
            ☰
          </button>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={newSession}
            aria-label="新建会话"
            title="新建会话"
          >
            ＋
          </button>
          <Link to="/" className="sidebar-toggle" aria-label="返回主页" title="返回主页">
            ←
          </Link>
        </div>
        <Outlet />

        {/* 分享对话框渲染在 .main-content 里，**不在 .sidebar 里** —— 理由见
            ShareDialog.tsx 顶部。它放在 <Outlet/> 之后，所以同层级的浮层一定
            画在页面内容之上（同 z-index 时后者居上）。 */}
        {shareId && <ShareDialog sessionId={shareId} onClose={() => setShareId(null)} />}
      </main>
    </div>
  );
}
