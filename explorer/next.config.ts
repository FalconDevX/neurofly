import type { NextConfig } from "next";

// Statyczny eksport: `npm run build` → out/ (dane z public/data ładowane fetch-em w przeglądarce).
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
