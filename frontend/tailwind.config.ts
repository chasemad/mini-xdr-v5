import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'], // Enable class-based dark mode for next-themes
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}'
  ],
  safelist: [
    // Dynamic severity colors used in various components
    { pattern: /^(bg|text|border)-(red|orange|yellow|green|blue|purple|gray|slate|zinc|neutral|stone)-(50|100|200|300|400|500|600|700|800|900)(\/(10|20|30|40|50|60|70|80|90))?$/, variants: ['hover', 'focus', 'disabled'] },
    // Dynamic agent type colors
    { pattern: /^(bg|text|border)-(blue|purple|green)-(50|100|200|300|400|500|600|700|800|900)(\/(10|20|30|40|50|60|70|80|90))?$/, variants: ['hover', 'focus'] },
  ],
  theme: {
    extend: {
      colors: {
        // Background colors
        bg: 'var(--bg)',
        surface: {
          0: 'var(--surface-0)',
          1: 'var(--surface-1)',
          2: 'var(--surface-2)',
        },

        // Text colors
        text: {
          DEFAULT: 'var(--text)',
          muted: 'var(--text-muted)',
          subtle: 'var(--text-subtle)',
        },

        // Border
        border: 'var(--border)',

        // Primary action color
        primary: 'var(--primary)',

        // Status colors
        info: 'var(--info)',
        success: 'var(--success)',
        warning: 'var(--warning)',
        danger: 'var(--danger)',

        // Severity scale for incidents
        severity: {
          info: 'var(--severity-info)',
          low: 'var(--severity-low)',
          med: 'var(--severity-med)',
          high: 'var(--severity-high)',
          critical: 'var(--severity-critical)',
        },

        // Interactive states
        hover: 'var(--hover)',
        active: 'var(--active)',
        focus: 'var(--focus)',
        disabled: 'var(--disabled)',
      },

      borderRadius: {
        xl: 'var(--radius-xl)',
        '2xl': 'var(--radius-2xl)',
        '3xl': 'var(--radius-3xl)',
      },

      boxShadow: {
        sm: 'var(--shadow-sm)',
        md: 'var(--shadow-md)',
        lg: 'var(--shadow-lg)',
        xl: 'var(--shadow-xl)',
        card: 'var(--shadow-card)',
      },

      fontFamily: {
        sans: 'var(--font-family-sans)',
        mono: 'var(--font-family-mono)',
      },

      spacing: {
        13: 'var(--spacing-5)',
        18: 'var(--spacing-8)',
        22: 'var(--spacing-12)',
      },

      transitionDuration: {
        0: '0ms',
        150: '150ms',
        200: '200ms',
        300: '300ms',
      },

      transitionTimingFunction: {
        'standard': 'cubic-bezier(0.25, 0.1, 0.25, 1)',
        'decelerate': 'cubic-bezier(0, 0, 0.2, 1)',
        'accelerate': 'cubic-bezier(0.4, 0, 1, 1)',
      },

      zIndex: {
        dropdown: 'var(--z-dropdown)',
        sticky: 'var(--z-sticky)',
        banner: 'var(--z-banner)',
        overlay: 'var(--z-overlay)',
        modal: 'var(--z-modal)',
        popover: 'var(--z-popover)',
        tooltip: 'var(--z-tooltip)',
      },

      screens: {
        sm: '640px',
        md: '768px',
        lg: '1024px',
        xl: '1280px',
        '2xl': '1536px',
      },
    },
  },
  plugins: [],
}

export default config
