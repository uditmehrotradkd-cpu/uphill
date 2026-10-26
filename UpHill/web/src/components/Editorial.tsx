import { CSSProperties, ReactNode } from "react";

/** Words slide up from behind a mask, staggered (editorial headline reveal). */
export function Reveal({ text, delay = 0, className = "" }: { text: string; delay?: number; className?: string }) {
  return (
    <span className={`reveal ${className}`}>
      {text.split(" ").map((word, i) => (
        <span key={i} className="reveal-mask">
          <span className="reveal-word" style={{ animationDelay: `${delay + i * 70}ms` } as CSSProperties}>{word}&nbsp;</span>
        </span>
      ))}
    </span>
  );
}

interface HeroProps {
  index: string;
  label: string;
  title: string;
  accent: string;
  outline: string;
  children?: ReactNode;
}

/** Page header: mono eyebrow, bold title with a serif-italic accent, giant outlined backdrop word. */
export function PageHero({ index, label, title, accent, outline, children }: HeroProps) {
  return (
    <header className="page-hero">
      <span className="hero-outline" aria-hidden>{outline}</span>
      <div className="eyebrow"><span>{index}</span> — {label}</div>
      <div className="hero-row">
        <h1 className="hero-title">
          <Reveal text={title} />
          <em><Reveal text={accent} delay={title.split(" ").length * 70 + 80} /></em>
        </h1>
        {children && <div className="hero-aside">{children}</div>}
      </div>
    </header>
  );
}
