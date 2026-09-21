import { Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ChatGate from './components/ChatGate';
import AuthFlow from './components/AuthFlow';
import ChatLayout from './components/ChatLayout';
import ChatPage from './pages/ChatPage';
import SharePage from './pages/SharePage';

/**
 * /login：走完两道门回主页。
 *
 * 登录完回主页而不是回 /chat —— 这个入口在主页导航上，用户的意图是「登录」，
 * 不是「进问答」；想进问答主页上另有一个按钮。回跳目标是常量，所以这里是
 * 唯一可以安全用 <Navigate> 的地方（AuthFlow.tsx 里说明了为什么别处不行）。
 */
function LoginRoute() {
  return (
    <AuthFlow>
      <Navigate to="/" replace />
    </AuthFlow>
  );
}

export default function App() {
  return (
    <Routes>
      {/* 公开主页：不经过任何鉴权 */}
      <Route path="/" element={<HomePage />} />

      {/* 问答应用：访问密码 + 患者身份两道门，均由 ChatGate 原地渲染 */}
      <Route element={<ChatGate />}>
        <Route path="/chat" element={<ChatLayout />}>
          <Route index element={<ChatPage />} />
        </Route>
      </Route>

      {/* 登录入口：从主页导航过来，两道门走完回主页。
          必须放在 ChatGate **外面** —— 它自己就是门，套在门里会变成
          「先登录才能看到登录页」。也必须在 path="*" 之前，否则被兜底吃掉。 */}
      <Route path="/login" element={<LoginRoute />} />

      {/* 兼容旧书签：历史记录页已删（侧栏会话列表能直接切换会话，独立的历史页
          只是同一份数据的第二个入口）。/chat/history 也一并失效 —— 它当初就是
          /history 的重定向目标，没有对外发布过，不必再留一层。
          必须放在 ChatGate **外面** —— 放里面会让未登录用户在错误的 URL 上
          看到登录页，等于让 URL 归一化依赖登录状态。 */}
      <Route path="/history" element={<Navigate to="/chat" replace />} />

      {/* 公开分享页：拿到链接即可看，不经过 ChatGate。
          位置是承重的，两条约束缺一不可：
            1. 必须在 ChatGate **外面** —— 放在里面会让访问者撞上访问密码门。
            2. 必须在下面的 path="*" **之前** —— 否则会被兜底重定向到 / 吃掉，
               表现为「链接打开后跳到主页」，看起来像分享功能坏了。 */}
      <Route path="/s/:token" element={<SharePage />} />

      {/* 兜底：未知路径回主页。
          这一条不是可选的：改动前任何 URL 都渲染 LoginPage，所以未匹配也看不出
          问题；主页上线后，未匹配的路径渲染 null，就是一片白屏。 */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
