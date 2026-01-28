import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { OutputLine } from '@/stores/agents';

interface MessageBlockProps {
  line: OutputLine;
}

export function MessageBlock({ line }: MessageBlockProps) {
  const { source, text } = line;

  // Determine styling based on source
  const isTool = source === 'tool';
  const isSystem = source === 'system';
  const isUser = source === 'user';

  const containerClass = isUser
    ? 'bg-primary/10 border-primary/20'
    : isSystem
      ? 'bg-warning/10 border-warning/20'
      : isTool
        ? 'bg-surface border-border'
        : 'bg-transparent border-transparent';

  const labelClass = isUser
    ? 'text-primary'
    : isSystem
      ? 'text-warning'
      : isTool
        ? 'text-secondary'
        : 'text-muted';

  const labelText = isUser
    ? 'You'
    : isSystem
      ? 'System'
      : isTool
        ? 'Tool'
        : 'Assistant';

  return (
    <div className={`rounded-lg border p-4 ${containerClass}`}>
      {/* Source label */}
      <div className={`mb-2 text-xs font-medium uppercase tracking-wider ${labelClass}`}>
        {labelText}
        {isTool && source !== 'tool' && (
          <span className="ml-2 font-normal normal-case text-muted">
            ({source})
          </span>
        )}
      </div>

      {/* Content */}
      <div className="prose prose-sm max-w-none dark:prose-invert">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Custom code block rendering
            code({ className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              const isInline = !match;

              if (isInline) {
                return (
                  <code
                    className="rounded bg-base px-1.5 py-0.5 text-sm"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }

              return (
                <pre className="overflow-x-auto rounded-lg bg-base p-4">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              );
            },
            // Custom table styling
            table({ children }) {
              return (
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse border border-border">
                    {children}
                  </table>
                </div>
              );
            },
            th({ children }) {
              return (
                <th className="border border-border bg-surface px-3 py-2 text-left">
                  {children}
                </th>
              );
            },
            td({ children }) {
              return (
                <td className="border border-border px-3 py-2">{children}</td>
              );
            },
            // Links
            a({ href, children }) {
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
    </div>
  );
}
