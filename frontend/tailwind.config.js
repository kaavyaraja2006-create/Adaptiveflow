/** @type {import('tailwindcss').Config} */
function withOpacity(variable) {
  return ({ opacityValue }) => {
    if (opacityValue !== undefined) return `rgb(var(${variable}) / ${opacityValue})`;
    return `rgb(var(${variable}))`;
  };
}

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: withOpacity("--color-ink"),
          light: withOpacity("--color-ink-light"),
          border: withOpacity("--color-ink-border"),
        },
        paper: withOpacity("--color-paper"),
        surface: withOpacity("--color-surface"),
        line: withOpacity("--color-line"),
        text: {
          DEFAULT: withOpacity("--color-text"),
          muted: withOpacity("--color-text-muted"),
        },
        brand: {
          DEFAULT: withOpacity("--color-brand"),
          dim: withOpacity("--color-brand-dim"),
          deep: withOpacity("--color-brand-deep"),
        },
        success: withOpacity("--color-success"),
        warning: withOpacity("--color-warning"),
        danger: withOpacity("--color-danger"),
      },
      fontFamily: {
        display: ["Sora", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: { xl2: "10px" },
    },
  },
  plugins: [],
}
