/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        obsidian: '#141518',
        panel: '#1a1b1f',
        accent: '#f5a623'
      }
    },
  },
  plugins: [],
}
