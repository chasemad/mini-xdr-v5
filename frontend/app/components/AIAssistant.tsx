"use client";

import { useState } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm your XDR AI Assistant. I can help you analyze incidents, manage blocking rules, and answer questions about your security environment. What would you like to know?",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      // Simulate AI response for now - this would call your MCP server
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const aiResponse: Message = {
        role: "assistant",
        content: getAIResponse(userMessage.content),
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error("Failed to get AI response:", error);
      
      const errorMessage: Message = {
        role: "assistant",
        content: "I'm sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const getAIResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();
    
    if (input.includes("status") || input.includes("health")) {
      return "Your XDR system is currently healthy. I can see you have 5 total incidents with several open cases. The auto-contain feature is disabled. Would you like me to analyze any specific incidents or enable auto-blocking?";
    }
    
    if (input.includes("block") || input.includes("contain")) {
      return "I can help you block threatening IPs. You currently have incidents from several IPs that may need containment. Would you like me to recommend which IPs to block based on their threat level and activity patterns?";
    }
    
    if (input.includes("unblock") || input.includes("allow")) {
      return "I can help you unblock IPs that are currently contained. Please let me know which IP address you'd like to unblock, or I can show you all currently blocked IPs for review.";
    }
    
    if (input.includes("incident") || input.includes("threat")) {
      return "I can analyze your incidents in detail. Your most recent incidents show SSH brute-force attacks from various IPs. Would you like me to provide a detailed analysis of any specific incident or recommend actions?";
    }
    
    if (input.includes("help") || input.includes("what can you do")) {
      return `I can help you with:
      
• **Incident Analysis**: Detailed threat assessment and recommendations
• **IP Management**: Block/unblock threatening IP addresses  
• **System Status**: Real-time monitoring and health checks
• **Security Recommendations**: Automated threat response suggestions
• **Historical Analysis**: Pattern detection across incidents

What would you like assistance with?`;
    }
    
    return "I understand you're asking about your XDR system. Could you be more specific? I can help with incident analysis, IP blocking/unblocking, system status, or security recommendations.";
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 h-14 w-14 bg-indigo-600 text-white rounded-full shadow-lg hover:bg-indigo-700 transition-colors z-50 flex items-center justify-center"
      >
        {isOpen ? (
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        )}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-96 bg-white border border-gray-200 rounded-xl shadow-2xl z-40 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-gray-200 bg-indigo-600 text-white rounded-t-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 bg-green-400 rounded-full"></div>
                <h3 className="font-semibold">XDR AI Assistant</h3>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-indigo-200 hover:text-white"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-xs p-3 rounded-xl text-sm ${
                    message.role === "user"
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  <p className={`text-xs mt-1 ${
                    message.role === "user" ? "text-indigo-200" : "text-gray-500"
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-900 max-w-xs p-3 rounded-xl text-sm">
                  <div className="flex items-center gap-2">
                    <div className="flex space-x-1">
                      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                      <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                    </div>
                    <span className="text-gray-500">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about incidents, blocking, or system status..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
