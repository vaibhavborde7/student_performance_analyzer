import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Student Performance Analyzer",
  description: "Machine learning dashboard for student performance analysis",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
