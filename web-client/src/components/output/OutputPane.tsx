import { useRef, useEffect } from 'react';
import { useAgentOutput } from '@/stores/agents';
import { MessageBlock } from './MessageBlock';

interface OutputPaneProps {
  agentId: string;
}

export function OutputPane({ agentId }: OutputPaneProps) {
  const output = useAgentOutput(agentId);
  const containerRef = useRef<HTMLDivElement>(null);
  const shouldAutoScroll = useRef(true);

  // Auto-scroll to bottom on new content
  useEffect(() => {
    if (shouldAutoScroll.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [output]);

  // Detect manual scroll to disable auto-scroll
  const handleScroll = () => {
    if (!containerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
    shouldAutoScroll.current = isAtBottom;
  };

  if (output.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <div className="mb-4 text-6xl opacity-20">
            <svg
              className="mx-auto h-16 w-16"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </div>
          <h2 className="mb-2 text-lg font-medium text-muted">
            Start a conversation
          </h2>
          <p className="text-sm text-muted">
            Type a message below to begin
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-4"
      onScroll={handleScroll}
    >
      <div className="mx-auto max-w-4xl space-y-4">
        {output.map((line) => (
          <MessageBlock key={line.id} line={line} />
        ))}
      </div>
    </div>
  );
}
