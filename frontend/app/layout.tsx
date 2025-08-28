import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import AIAssistant from "./components/AIAssistant";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mini-XDR",
  description: "SSH Brute-Force Detection and Response System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          {/* Navigation */}
          <nav className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex items-center">
                  <Link href="/" className="text-xl font-bold text-gray-900">
                    Mini-XDR
                  </Link>
                </div>
                <div className="flex items-center space-x-6">
                  <Link 
                    href="/" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Overview
                  </Link>
                  <Link 
                    href="/incidents" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Incidents
                  </Link>
                  <Link 
                    href="/hunt" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Hunt
                  </Link>
                  <Link 
                    href="/investigations" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Investigations
                  </Link>
                  <Link 
                    href="/intelligence" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Intelligence
                  </Link>
                  <Link 
                    href="/agents" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Agents
                  </Link>
                  <Link 
                    href="/analytics" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Analytics
                  </Link>
                  <Link 
                    href="/settings" 
                    className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Settings
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Main content */}
          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {children}
          </main>
          
          {/* AI Assistant */}
          <AIAssistant />
        </div>
      </body>
    </html>
  );
}
