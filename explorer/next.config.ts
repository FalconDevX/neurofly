import type { NextConfig } from "next";

// Statyczny eksport: `npm run build` → out/ (dane z public/data ładowane fetch-em w przeglądarce).
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // "." tylko w buildzie pod Artifact (scripts/build-artifact.mjs); zwykły build bez zmian
  assetPrefix: process.env.ASSET_PREFIX,
};

export default nextConfig;
