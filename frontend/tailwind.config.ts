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
        // Shadcn/UI colors
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },

      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
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
