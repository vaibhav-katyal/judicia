import { useEffect, useRef, type FormEvent, type RefObject } from "react";
import { ArrowRightIcon, SearchIcon } from "@/components/Icons";

type ComposerProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
  compact?: boolean;
  autoFocus?: boolean;
  textareaRef?: RefObject<HTMLTextAreaElement | null>;
};

export function Composer({
  value,
  onChange,
  onSubmit,
  loading = false,
  compact = false,
  autoFocus = false,
  textareaRef,
}: ComposerProps) {
  const localRef = useRef<HTMLTextAreaElement>(null);
  const ref = textareaRef ?? localRef;

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, compact ? 150 : 200)}px`;
  }, [value, compact, ref]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!value.trim() || loading) return;
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div
        className={`surface-plate group rounded-2xl bg-card transition-all duration-300 focus-within:border-accent/80 focus-within:shadow-[0_0_0_4px_color-mix(in_oklab,var(--brass)_14%,transparent)] ${
          compact ? "p-2" : "p-3 sm:p-4"
        }`}
      >
        <div className="flex items-start gap-3">
          <SearchIcon
            className={`mt-3 h-5 w-5 shrink-0 text-muted-foreground transition-colors group-focus-within:text-accent ${
              compact ? "ml-1" : "ml-1 sm:ml-2"
            }`}
          />
          <textarea
            ref={ref}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                if (value.trim() && !loading) onSubmit();
              }
            }}
            rows={compact ? 2 : 3}
            maxLength={2000}
            autoFocus={autoFocus}
            aria-label="Describe your legal situation"
            placeholder={
              compact
                ? "Ask about another legal problem…"
                : "Describe what happened. For example: “Someone hacked my Instagram and changed my password. They also tried to take money from my bank account.”"
            }
            className={`min-h-[72px] w-full resize-none bg-transparent py-2.5 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground/80 ${
              compact ? "px-1" : "px-1 sm:text-base"
            }`}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/70 px-1 pt-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
            <span className="font-mono tabular-nums">{value.length}/2000</span>
            {!compact && (
              <>
                <span className="hidden h-3 w-px bg-border sm:block" />
                <span className="hidden sm:inline">
                  <kbd className="rounded border border-border bg-secondary px-1.5 py-0.5 font-mono text-[10px]">
                    Ctrl
                  </kbd>{" "}
                  +{" "}
                  <kbd className="rounded border border-border bg-secondary px-1.5 py-0.5 font-mono text-[10px]">
                    Enter
                  </kbd>{" "}
                  to send
                </span>
              </>
            )}
          </div>

          <button
            type="submit"
            disabled={!value.trim() || loading}
            aria-busy={loading}
            className={`ink-panel inline-flex items-center justify-center gap-2 rounded-xl px-5 text-sm font-semibold tracking-wide transition-all duration-200 hover:brightness-125 disabled:cursor-not-allowed disabled:opacity-45 ${
              compact ? "h-10" : "h-11 px-6"
            }`}
          >
            {loading ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Building plan…
              </>
            ) : (
              <>
                {compact ? "Ask Judicia" : "Get my first-aid plan"}
                <ArrowRightIcon className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {!compact && (
        <p className="mt-3 px-1 text-center text-[11px] leading-relaxed text-muted-foreground">
          Include dates, amounts, and what you want to happen. Do not share passwords,
          OTPs, or full bank details.
        </p>
      )}
    </form>
  );
}
