import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

// Geist jak w shadcn/ui, jeden krój do nagłówków i treści
const geist = Geist({ subsets: ["latin", "latin-ext"], variable: "--font-body" });
const mono = Geist_Mono({ subsets: ["latin", "latin-ext"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "NeuroFly BANC Explorer",
  description: "Connectome BANC v888 w 3D z renderem na GPU: somy, szkielety neuronów lotu, partnerzy synaptyczni i komendy drona.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl" className={`${geist.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
