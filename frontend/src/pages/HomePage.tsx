import { Link } from 'react-router-dom';
import {
  isAuthenticated,
  hasPatient,
  getToken,
  getPatientToken,
  getPatientLabel,
  clearToken,
  clearPatientId,
} from '../api/config';

/** 卡片状态用字符串字面量而非 enum：tsconfig 的 erasableSyntaxOnly 禁止 enum。 */
type HomeCardStatus = 'live' | 'planned';

interface HomeCard {
  key: string;
  icon: string;
  title: string;
  description: string;
  tags: string[];
  status: HomeCardStatus;
  /** status 为 live 时的站内路由。站外卡片改用 external，两者只填一个。 */
  to?: string;
  /**
   * 站外应用（子域名）的源。填了这个就渲染成 <a target="_blank">，点击时还要
   * 把当前令牌用 URL fragment 交接过去 —— www 与 lit 是两个源，localStorage
   * 不互通，登录态必须显式传递。站内卡片不要填。
   */
  external?: string;
}

/**
 * 站外应用的源，构建时由 VITE_ACADEMIC_URL 注入。
 *
 * **留空则下面那张卡片整张不出现**，而不是出现一张点不动的卡片：主站与学术
 * 工作台是两个独立部署，源码里写死一个私有地址，对克隆者就是一个死链接。
 * 自己的部署在 frontend/.env 里填上即可恢复，那一行不会进版本控制。
 */
const ACADEMIC_URL = import.meta.env.VITE_ACADEMIC_URL as string | undefined;

const ACADEMIC_CARD: HomeCard = {
  key: 'academic',
  icon: '📚',
  title: 'AI 学术工作台',
  description: '面向科研场景的文献检索、综述生成与实验数据整理解读。',
  tags: ['文献检索', '综述生成'],
  status: 'live',
  external: ACADEMIC_URL,
};

/**
 * 加一个新功能入口 = 往这个数组里加一个对象，渲染循环不用动。
 */
const CARDS: HomeCard[] = [
  {
    key: 'qa',
    icon: '🩺',
    title: '智能问答系统',
    description:
      '基于图谱增强检索（GraphRAG）的医学知识问答。融合向量语义检索、知识图谱推理与对话记忆，支持多轮追问与推理路径回溯。',
    tags: ['GraphRAG', '向量检索', '知识图谱', '对话记忆'],
    status: 'live',
    to: '/chat',
  },
  ...(ACADEMIC_URL ? [ACADEMIC_CARD] : []),
];

/**
 * 拼接交给站外应用的地址：`#a=<访问令牌>&p=<患者令牌>&l=<账号名>`。
 *
 * 用 fragment 而不是 query string，是因为 fragment 不发给服务器、不进 Referer、
 * 跨源可读。代价是在对方 replaceState 之前它留在浏览器历史里，所以那边 mount
 * 时会同步抹掉。l 只用于界面展示（截图里显示账号名），不参与任何校验。
 */
function handoffUrl(origin: string): string {
  const parts = [`a=${encodeURIComponent(getToken())}`];
  const patient = getPatientToken();
  if (patient) parts.push(`p=${encodeURIComponent(patient)}`);
  const label = getPatientLabel();
  if (label) parts.push(`l=${encodeURIComponent(label)}`);
  return `${origin}/#${parts.join('&')}`;
}

const CAPABILITIES = [
  {
    title: '混合检索',
    body: '向量语义检索与知识图谱推理并行，再由路由层按问题复杂度选择策略。',
  },
  {
    title: '多轮记忆',
    body: '对话上下文按患者隔离保存，跨会话召回相关历史，支持追问与指代消解。',
  },
  {
    title: '可追溯',
    body: '每次回答都会展示命中的检索策略与推理依据，而不是一个黑盒结论。',
  },
];

export default function HomePage() {
  /**
   * 登录态。直接读、不放进 state —— 这里没有需要订阅的变更源：
   * 登录（/login → Navigate 回 /）和退出（整页 assign）都会让本组件重新 mount，
   * 重挂时这个表达式自然会重算。用 effect 去同步反而会踩本仓库
   * `react-hooks/set-state-in-effect` 那条 error。
   *
   * label 为空时退回「已登录」：账号体系下 label 是注册填的昵称或邮箱，
   * 但别让一个空的 localStorage 键把导航渲染成一片空白。
   */
  const account =
    hasPatient() && isAuthenticated() ? getPatientLabel() || '已登录' : '';

  const logout = () => {
    // 与 ChatLayout 的「退出登录」保持一致：两个令牌一起清再整页重载。少清一个
    // 会留下「已登出但卡片还带着旧令牌跳去 lit」这种半登出状态。
    clearToken();
    clearPatientId();
    window.location.reload();
  };

  return (
    <div className="home-page">
      <nav className="home-nav">
        <Link to="/" className="home-brand">
          智能问答系统
        </Link>
        <div className="home-nav-links">
          <a className="home-nav-link" href="#products">
            产品
          </a>
          <a className="home-nav-link" href="#about">
            关于
          </a>
          {account ? (
            <>
              <span className="home-nav-account" title={account}>
                {account}
              </span>
              <button
                type="button"
                className="home-nav-link home-nav-btn"
                onClick={logout}
              >
                退出登录
              </button>
            </>
          ) : (
            <Link className="home-nav-link" to="/login">
              登录
            </Link>
          )}
          <Link className="home-nav-link home-nav-enter" to="/chat">
            进入系统 →
          </Link>
        </div>
      </nav>

      <header className="home-hero">
        <p className="home-eyebrow">GraphRAG · 知识图谱增强检索</p>
        <h1 className="home-title">
          把知识图谱接进
          <br />
          大模型的问答系统
        </h1>
        <p className="home-lede">
          面向医学知识库的检索增强问答：向量检索负责语义召回，知识图谱负责关系推理，
          路由层决定每个问题该走哪条路。回答附带推理依据，可逐条追溯。
        </p>
        <div className="home-actions">
          <Link className="home-btn home-btn-primary" to="/chat">
            进入问答系统 →
          </Link>
          <a className="home-btn home-btn-ghost" href="#products">
            了解功能
          </a>
        </div>
      </header>

      <section className="home-section" id="products">
        <div className="home-section-head">
          <h2 className="home-section-title">产品</h2>
          <span className="home-section-note">更多工具正在开发中</span>
        </div>

        <div className="home-grid">
          {CARDS.map((card) => {
            const body = (
              <>
                <div className="home-card-icon">{card.icon}</div>
                <h3 className="home-card-title">{card.title}</h3>
                <p className="home-card-desc">{card.description}</p>
                <div className="home-card-tags">
                  {card.tags.map((t) => (
                    <span key={t} className="home-tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            );

            if (card.status !== 'live') {
              // 占位卡片用 div + aria-disabled，不用 disabled <button>（会被移出
              // tab 序却仍被读作控件），也不用 <Link to="#">（会真的导航并滚到顶）。
              return (
                <div
                  key={card.key}
                  className="home-card home-card-planned"
                  aria-disabled="true"
                >
                  <span className="home-badge-soon">即将上线</span>
                  {body}
                </div>
              );
            }

            const cta = <span className="home-card-cta">进入 →</span>;

            // 站外卡片必须用 <a>：<Link> 是站内路由，指向子域名只会得到一次
            // 错误的导航。
            if (card.external) {
              // 未登录时不把人送去子域名 —— 那边没有 fragment 可收，只会看到
              // 一屏「请回主站登录」。先在本站把门过完。
              return account ? (
                <a
                  key={card.key}
                  className="home-card home-card-live"
                  href={handoffUrl(card.external)}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {body}
                  {cta}
                </a>
              ) : (
                <Link
                  key={card.key}
                  to="/login"
                  className="home-card home-card-live"
                >
                  {body}
                  {cta}
                </Link>
              );
            }

            return (
              <Link
                key={card.key}
                to={card.to || '/'}
                className="home-card home-card-live"
              >
                {body}
                {cta}
              </Link>
            );
          })}
        </div>
      </section>

      <section className="home-section" id="about">
        <div className="home-section-head">
          <h2 className="home-section-title">关于这个系统</h2>
        </div>
        <div className="home-about">
          {CAPABILITIES.map((c) => (
            <div key={c.title} className="home-capability">
              <h3 className="home-capability-title">{c.title}</h3>
              <p className="home-capability-body">{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="home-footer">
        <p className="home-footer-note">
          本系统输出由模型生成，仅供研究与技术演示参考，不构成医疗建议。
        </p>
      </footer>
    </div>
  );
}
