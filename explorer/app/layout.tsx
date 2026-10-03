import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

// Geist jak w shadcn/ui, jeden krój do nagłówków i treści
const geist = Geist({ subsets: ["latin", "latin-ext"], variable: "--font-body" });
const mono = Geist_Mono({ subsets: ["latin", "latin-ext"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "NeuroFly BANC Explorer",
  icons: { icon: "logo-fly.png" },
  description: "BANC v888 connectome in 3D, rendered on the GPU: somas, flight-neuron skeletons, synaptic partners and drone commands.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
