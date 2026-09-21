import { Outlet } from 'react-router-dom';
import AuthFlow from './AuthFlow';

/**
 * 守卫问答应用（/chat 及其子路由）。主页 / 是公开的，不经过这里。
 *
 * 两道门的语义、「原地渲染而不是重定向」的原因，都在 AuthFlow.tsx 里 ——
 * 那段注释跟着代码走了，不要在这里复述一份会漂移的副本。
 */
export default function ChatGate() {
  return (
    <AuthFlow>
      <Outlet />
    </AuthFlow>
  );
}
