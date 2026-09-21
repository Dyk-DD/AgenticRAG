import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';

export default function ChatPage() {
  const { messages, loading, streaming, error, sendMessage, stopStreaming } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  const isEmpty = messages.length === 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className={`chat-page ${isEmpty ? 'chat-page-empty' : ''}`}>
      <div className="chat-messages">
        {isEmpty && (
          <div className="chat-welcome">
            <h2 className="chat-welcome-title">您好！我是您的智能问答助手。</h2>
            <p className="chat-welcome-sub">您可以向我咨询病症初筛、药物禁忌或复杂的临床关联问题。</p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage
            key={msg.id}
            message={msg}
            // 引用相等判定「最后一条」—— store 里必须保持不可变更新，改成
            // 就地 mutate 会让所有历史助手消息一起进入 streaming 态。
            streaming={streaming && msg.role === 'assistant' && msg === messages[messages.length - 1]}
          />
        ))}

        {error && <div className="error-banner">⚠️ {error}</div>}

        <div ref={bottomRef} />
      </div>

      {/* ChatInput 必须始终挂在同一位置。空态与有消息态若各渲染一个，发出第一条
          消息时组件会重挂载，输入框丢焦点、用户得重新点一次才能接着打字。
          所以空态居中的实现绕开了它：问候语留在 .chat-messages 里，示例按钮
          排在它之后，由 .chat-page-empty 的 justify-content 把三块整体居中。 */}
      <ChatInput
        onSend={sendMessage}
        onStop={stopStreaming}
        streaming={streaming}
        disabled={loading && !streaming}
      />

      {isEmpty && (
        <div className="example-questions">
          <button type="button" onClick={() => sendMessage('高血压合并糖尿病患者的饮食禁忌有哪些？')}>
            高血压合并糖尿病的饮食禁忌
          </button>
          <button type="button" onClick={() => sendMessage('冠心病患者使用阿司匹林的注意事项？')}>
            冠心病阿司匹林注意事项
          </button>
        </div>
      )}
    </div>
  );
}
