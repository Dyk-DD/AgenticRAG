import { useState, type ReactNode } from 'react';
import LoginPage from './LoginPage';
import AccountGate from './AccountGate';
import { isAuthenticated, hasPatient } from '../api/config';

interface Props {
  /** 两道门都过了才渲染。门后放什么由调用方决定（问答界面、主页回跳……）。 */
  children: ReactNode;
}

/**
 * 两道门：访问密码 → 账号 → 内容（children）。
 *
 * 从 ChatGate 里抽出来是为了让 /login 也能走同一条链，所以它把「门后是什么」
 * 交给调用方，而不是自己渲染 <Outlet />。
 *
 * 登录页与账号门都「原地渲染」而不是重定向，这不是风格选择：
 * api/client.ts 在收到 401/403 时会 clearToken() + clearPatientId() + reload，
 * 刷新保留当前 URL，所以未登录状态必须在同一个 URL 上被接住。改成
 * <Navigate to="/"> 会把用户在会话中途弹到主页且不作解释。
 *
 * /login 这条路由**不是**上面那条约束的反例：它不需要记住「登录后回到哪」
 * （它只从主页导航进来，回跳目标恒为 /），而且 401 那条路径永远不经过它 ——
 * reload 回的是用户当时所在的 URL，由 ChatGate 原地接住。两者互不冲突。
 */
export default function AuthFlow({ children }: Props) {
  const [authed, setAuthed] = useState(isAuthenticated());
  const [patientReady, setPatientReady] = useState(hasPatient());

  if (!authed) {
    return <LoginPage onLogin={() => setAuthed(true)} />;
  }
  if (!patientReady) {
    return <AccountGate onReady={() => setPatientReady(true)} />;
  }
  return <>{children}</>;
}
