import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#09090b",
        foreground: "#fafafa",
        card: "#111113",
        muted: "#27272a",
        border: "#3f3f46",
        primary: "#2563eb"
      }
    }
  },
  plugins: []
}

export default config
