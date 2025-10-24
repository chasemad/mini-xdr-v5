"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { register } from "../lib/api";
import { useAuth } from "../contexts/AuthContext";

export default function RegisterPage() {
  const router = useRouter();
  const { login, isAuthenticated, loading: authLoading } = useAuth();
  const [formData, setFormData] = useState({
    organization_name: "",
    admin_email: "",
    admin_password: "",
    admin_name: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Redirect if already authenticated
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.push("/");
    }
  }, [authLoading, isAuthenticated, router]);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Register organization
      const data = await register(formData);
      
      // Store tokens
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("refresh_token", data.refresh_token);
      
      // Login to populate auth context
      await login(formData.admin_email, formData.admin_password);
      
      // Login function handles redirect
    } catch (err: any) {
      setError(err.message || "Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-black py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 p-10 bg-gray-800 rounded-xl shadow-2xl border border-gray-700">
        <div>
          <h2 className="mt-6 text-center text-4xl font-extrabold text-white">
            Create Organization
          </h2>
          <p className="mt-2 text-center text-sm text-gray-400">
            Register a new organization and admin account
          </p>
        </div>
        
        <form className="mt-8 space-y-6" onSubmit={handleRegister}>
          {error && (
            <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded">
              {error}
            </div>
          )}
          
          <div className="space-y-4">
            <div>
              <label htmlFor="organization_name" className="block text-sm font-medium text-gray-300 mb-1">
                Organization Name
              </label>
              <input
                id="organization_name"
                name="organization_name"
                type="text"
                required
                className="appearance-none relative block w-full px-3 py-3 border border-gray-600 placeholder-gray-400 text-white bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-sm"
                placeholder="Acme Security Corp"
                value={formData.organization_name}
                onChange={(e) =>
                  setFormData({ ...formData, organization_name: e.target.value })
                }
              />
            </div>

            <div>
              <label htmlFor="admin_name" className="block text-sm font-medium text-gray-300 mb-1">
                Admin Full Name
              </label>
              <input
                id="admin_name"
                name="admin_name"
                type="text"
                required
                className="appearance-none relative block w-full px-3 py-3 border border-gray-600 placeholder-gray-400 text-white bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-sm"
                placeholder="John Doe"
                value={formData.admin_name}
                onChange={(e) =>
                  setFormData({ ...formData, admin_name: e.target.value })
                }
              />
            </div>

            <div>
              <label htmlFor="admin_email" className="block text-sm font-medium text-gray-300 mb-1">
                Admin Email
              </label>
              <input
                id="admin_email"
                name="admin_email"
                type="email"
                required
                className="appearance-none relative block w-full px-3 py-3 border border-gray-600 placeholder-gray-400 text-white bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-sm"
                placeholder="admin@example.com"
                value={formData.admin_email}
                onChange={(e) =>
                  setFormData({ ...formData, admin_email: e.target.value })
                }
              />
            </div>

            <div>
              <label htmlFor="admin_password" className="block text-sm font-medium text-gray-300 mb-1">
                Admin Password
              </label>
              <input
                id="admin_password"
                name="admin_password"
                type="password"
                required
                minLength={12}
                className="appearance-none relative block w-full px-3 py-3 border border-gray-600 placeholder-gray-400 text-white bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent sm:text-sm"
                placeholder="Min 12 chars, with uppercase, lowercase, number & special char"
                value={formData.admin_password}
                onChange={(e) =>
                  setFormData({ ...formData, admin_password: e.target.value })
                }
              />
              <p className="mt-1 text-xs text-gray-400">
                Must be at least 12 characters with uppercase, lowercase, number, and special character
              </p>
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Creating organization..." : "Create Organization"}
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div className="text-sm">
              <a
                href="/login"
                className="font-medium text-blue-400 hover:text-blue-300"
              >
                Already have an account? Sign in
              </a>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

