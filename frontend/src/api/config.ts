const TOKEN_KEY = 'agentic_rag_token';

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) || '';
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  return getToken() !== '';
}

// ── Patient identity ────────────────────────────────────────────────
// 身份凭据是服务端签发的患者令牌，不是患者 ID。ID 只用于界面展示，
// 请求时不再发送 —— 后端只认验签通过的令牌，改 ID 无法冒充他人。

const PATIENT_KEY = 'agentic_rag_patient';
const PATIENT_TOKEN_KEY = 'agentic_rag_patient_token';
const PATIENT_LABEL_KEY = 'agentic_rag_patient_label';

/** 仅用于界面展示（如账号徽章），不可作为身份凭据。 */
export function getPatientId(): string {
  return localStorage.getItem(PATIENT_KEY) || '';
}

export function setPatientId(pid: string): void {
  localStorage.setItem(PATIENT_KEY, pid);
}

/** 界面上显示的账号名（注册填的名字，或邮箱）。同样只用于展示。 */
export function getPatientLabel(): string {
  return localStorage.getItem(PATIENT_LABEL_KEY) || '';
}

export function setPatientLabel(label: string): void {
  localStorage.setItem(PATIENT_LABEL_KEY, label);
}

export function getPatientToken(): string {
  return localStorage.getItem(PATIENT_TOKEN_KEY) || '';
}

export function setPatientToken(token: string): void {
  localStorage.setItem(PATIENT_TOKEN_KEY, token);
}

// 账号里存的 id 形如 u_3f9a2b1c8d4e，人要读的是 label，所以两个一起清 ——
// 漏掉 label 的话徽章会退回显示那串 id。
export function clearPatientId(): void {
  localStorage.removeItem(PATIENT_KEY);
  localStorage.removeItem(PATIENT_TOKEN_KEY);
  localStorage.removeItem(PATIENT_LABEL_KEY);
}

export function hasPatient(): boolean {
  return getPatientToken() !== '';
}
