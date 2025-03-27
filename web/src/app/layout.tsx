import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Birdophile by Timotej",
  description: "We like them birds",
};
export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  minimumScale: 1,
  maximumScale: 1,
  userScalable: false,
};
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full overscroll-none">
        <head>
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta
          name="apple-mobile-web-app-status-bar-style"
          content="white-translucent"
        />
        <meta name="theme-color" content="#F45B2D" />
        <link rel="apple-touch-icon" href="/apple-icon.png" />
        
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased overscroll-none`} 
      >
      <main className="flex flex-col items-center justify-between min-h-screen">
        <div className="flex items-center justify-center w-full bg-tangerine bg-contain bg-left bg-no-repeat bg-[url('/oldman.png')] shadow-xl">
            <h1 className="text-3xl md:text-5xl uppercase tracking-widest pt-2 pb-32 md:pt-24 md:pb-24 font-bold text-medium ">Birdophile</h1>
          </div>
        {children}
        <footer className="flex items-center justify-center w-full py-4">
          <p className="text-sm text-tangerine">
            &copy; Timotej Drnovsek Pangerc 2025
          </p>
        </footer>
      </main>
      </body>
    </html>
  );
}
