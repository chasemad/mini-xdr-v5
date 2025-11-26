import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppProvider } from "./contexts/AppContext";
import { AuthProvider } from "./contexts/AuthContext";
import { DashboardProvider } from "./contexts/DashboardContext";
import { ToastProvider } from "@/components/ui/toast";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mini-XDR - SOC Command Center",
  description: "Enterprise Security Operations Center - Advanced Threat Detection and Response",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <AppProvider>
            <DashboardProvider>
              <ToastProvider>
                {children}
              </ToastProvider>
            </DashboardProvider>
          </AppProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
