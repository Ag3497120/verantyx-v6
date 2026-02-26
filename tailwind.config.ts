import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        electric: '#0EA5E9',
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { textShadow: '0 0 20px rgba(14, 165, 233, 0.5), 0 0 30px rgba(14, 165, 233, 0.3)' },
          '100%': { textShadow: '0 0 30px rgba(14, 165, 233, 0.8), 0 0 40px rgba(14, 165, 233, 0.5)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
