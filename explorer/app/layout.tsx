import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Sora } from "next/font/google";

import "./globals.css";

const sora = Sora({ subsets: ["latin", "latin-ext"], weight: ["500", "600", "700"], variable: "--font-display" });
const inter = Inter({ subsets: ["latin", "latin-ext"], variable: "--font-body" });
const mono = JetBrains_Mono({ subsets: ["latin", "latin-ext"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "NeuroFly BANC Explorer",
  description: "Connectome BANC v888 w 3D z renderem na GPU: somy, szkielety neuronów lotu, partnerzy synaptyczni i komendy drona.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl" className={`${sora.variable} ${inter.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
