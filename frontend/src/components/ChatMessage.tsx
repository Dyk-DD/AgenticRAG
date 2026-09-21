import type { Message } from '../types';
import RoutingCard from './RoutingCard';

interface Props {
  message: Message;
  streaming?: boolean;
}

export default function ChatMessage({ message, streaming }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      {/* 用户消息不带头像（DeepSeek 网页端如此，右侧气泡本身已经表明身份）。
          保留的这个是纯装饰，对屏幕阅读器隐藏，免得把「听诊器」念出来。 */}
      {!isUser && (
        <div className="message-avatar" aria-hidden="true">
          🩺
        </div>
      )}
      <div className="message-body">
        {/* 推理面板排在正文之前：它讲的是「这条回答是怎么来的」，读者该在读答案
            之前看到，而不是拉到最下面才发现。
            原先这里另有一行 .routing-mini 摘要，已删除 —— 它与本面板重复。 */}
        {message.routing && <RoutingCard routing={message.routing} />}
        <div className="message-text">
          {message.content || (streaming ? '思考中...' : '')}
          {streaming && <span className="cursor-blink">▌</span>}
        </div>
      </div>
    </div>
  );
}
