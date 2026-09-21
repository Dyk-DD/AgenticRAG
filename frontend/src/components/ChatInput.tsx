import { useState, useRef, useEffect } from 'react';

interface Props {
  onSend: (text: string) => void;
  /** 流式中断。streaming 为真时发送按钮换成停止按钮。 */
  onStop: () => void;
  streaming?: boolean;
  /** 只在「非流式的等待期」（如建会话）禁用输入。流式期间输入框保持可用 ——
      用户可以边看回答边打下一句。 */
  disabled?: boolean;
}

// 自适应高度的上限。这个值**不在 CSS 里**，改的时候别只翻样式表。
const MAX_HEIGHT = 150;

export default function ChatInput({ onSend, onStop, streaming, disabled }: Props) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, MAX_HEIGHT) + 'px';
    }
  }, [input]);

  const handleSend = () => {
    const trimmed = input.trim();
    // streaming 期间回车不发：此时按钮已经是「停止」，store 里的 sendMessage
    // 也有同样的守卫，这里只是让「按了没反应」这件事有个明确出处。
    if (!trimmed || disabled || streaming) return;
    onSend(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input-container">
      <textarea
        ref={textareaRef}
        className="chat-input"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="请输入症状、药物、科室或临床疑问..."
        rows={1}
        disabled={disabled}
      />
      {streaming ? (
        <button type="button" className="btn-stop" onClick={onStop} aria-label="停止生成" title="停止生成">
          ■
        </button>
      ) : (
        <button
          type="button"
          className="btn-send"
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          aria-label="发送"
          title="发送"
        >
          ↑
        </button>
      )}
    </div>
  );
}
