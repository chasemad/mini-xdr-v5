"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, HelpCircle, RotateCcw } from "lucide-react";
import { agentOrchestrate, confirmAgentAction } from "../lib/api";
import { ConfirmationPrompt } from "@/components/ConfirmationPrompt";
import { ExecutionResultDisplay } from "@/components/ExecutionResultDisplay";

interface Incident {
  id: number;
  created_at: string;
  src_ip: string;
  reason: string;
  status: string;
  auto_contained: boolean;
  triage_note?: {
    summary: string;
    severity: string;
    recommendation: string;
    rationale: string[];
  };
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai' | 'confirmation' | 'follow_up';
  content: string;
  timestamp: Date;
  loading?: boolean;
  // For confirmation messages
  actionPlan?: any;
  pendingActionId?: string;
  conversationId?: string;
  // For follow-up messages
  followUpQuestions?: string[];
  suggestedOptions?: Record<string, string[]>;
  // For execution results
  executionDetails?: any;
}

interface AIChatInterfaceProps {
  selectedIncident?: Incident | null;
}

export default function AIChatInterface({ selectedIncident }: AIChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "Welcome to the SOC Command Center. I'm your AI analyst assistant. I can help you investigate incidents, analyze threats, and coordinate response actions. What would you like to explore?",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [pendingActionId, setPendingActionId] = useState<string | undefined>();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Add loading message
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await agentOrchestrate(userMessage.content, selectedIncident?.id, {
        incident_data: selectedIncident,
        chat_history: messages.slice(-5),
        conversation_id: conversationId,
        pending_action_id: pendingActionId
      });

      // Update conversation state
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
      if (response.pending_action_id) {
        setPendingActionId(response.pending_action_id);
      } else {
        setPendingActionId(undefined);
      }

      // Handle different response types
      const responseType = response.response_type || 'answer';

      let newMessage: ChatMessage;

      if (responseType === 'confirmation_required') {
        newMessage = {
          id: (Date.now() + 2).toString(),
          type: 'confirmation',
          content: response.message || "Ready to execute?",
          timestamp: new Date(),
          actionPlan: response.action_plan,
          pendingActionId: response.pending_action_id,
          conversationId: response.conversation_id
        };
      } else if (responseType === 'follow_up') {
        newMessage = {
          id: (Date.now() + 2).toString(),
          type: 'follow_up',
          content: response.message || "I need more information.",
          timestamp: new Date(),
          followUpQuestions: response.follow_up_questions,
          suggestedOptions: response.suggested_options,
          conversationId: response.conversation_id
        };
      } else {
        newMessage = {
          id: (Date.now() + 2).toString(),
          type: 'ai',
          content: response.message || response.analysis || "I've processed your request.",
          timestamp: new Date(),
          executionDetails: response.execution_details
        };
      }

      setMessages(prev => prev.slice(0, -1).concat(newMessage));

    } catch (error) {
      setMessages(prev => prev.slice(0, -1));
      console.error('AI response failed:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: "I apologize, but I'm having trouble connecting to the AI service right now. Please try again in a moment.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmation = async (approved: boolean, msgPendingActionId?: string) => {
    if (!msgPendingActionId) return;

    setLoading(true);

    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    };

    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await confirmAgentAction(
        msgPendingActionId,
        approved,
        selectedIncident?.id,
        { incident_data: selectedIncident }
      );

      setPendingActionId(undefined);

      const newMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: response.message || (approved ? "Action approved and executed!" : "Action cancelled."),
        timestamp: new Date()
      };

      setMessages(prev => prev.slice(0, -1).concat(newMessage));

    } catch (error) {
      setMessages(prev => prev.slice(0, -1));
      console.error('Confirmation failed:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: "Failed to process confirmation. Please try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestedOption = (option: string) => {
    setInput(option);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleClearChat = () => {
    // Reset to initial welcome message
    setMessages([{
      id: '1',
      type: 'ai',
      content: "Welcome to the SOC Command Center. I'm your AI analyst assistant. I can help you investigate incidents, analyze threats, and coordinate response actions. What would you like to explore?",
      timestamp: new Date()
    }]);

    // Clear conversation state
    setConversationId(undefined);
    setPendingActionId(undefined);
    setInput('');
  };

  return (
    <div className="bg-surface-0 border border-border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold text-text">AI Analyst Assistant</h3>
            </div>
            <p className="text-sm text-text-muted mt-1">
              Ask questions about incidents, threats, or response strategies
            </p>
          </div>
          {messages.length > 1 && (
            <button
              onClick={handleClearChat}
              className="p-2 hover:bg-surface-1 rounded-lg transition-colors text-text-muted hover:text-text"
              title="New conversation"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 max-h-96 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => {
          // User message
          if (message.type === 'user') {
            return (
              <div key={message.id} className="flex gap-3 justify-end">
                <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-primary text-bg ml-12">
                  <p className="text-sm leading-relaxed">{message.content}</p>
                </div>
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                  <User className="w-4 h-4 text-bg" />
                </div>
              </div>
            );
          }

          // Confirmation message
          if (message.type === 'confirmation') {
            return (
              <div key={message.id} className="space-y-2">
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-primary" />
                  </div>
                  <div className="flex-1 max-w-[90%]">
                    <ConfirmationPrompt
                      actionPlan={message.actionPlan}
                      message={message.content}
                      onApprove={() => handleConfirmation(true, message.pendingActionId)}
                      onReject={() => handleConfirmation(false, message.pendingActionId)}
                      isLoading={loading}
                    />
                  </div>
                </div>
              </div>
            );
          }

          // Follow-up questions message
          if (message.type === 'follow_up') {
            return (
              <div key={message.id} className="space-y-2">
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                    <HelpCircle className="w-4 h-4 text-primary" />
                  </div>
                  <div className="flex-1 max-w-[85%]">
                    <div className="rounded-2xl px-4 py-3 bg-surface-1 text-text">
                      <p className="text-sm leading-relaxed">{message.content}</p>

                      {message.followUpQuestions && message.followUpQuestions.length > 0 && (
                        <div className="mt-3 space-y-2">
                          {message.followUpQuestions.map((question, idx) => (
                            <div key={idx} className="text-xs text-text-muted flex items-start gap-1">
                              <span className="mt-0.5">•</span>
                              <span>{question}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {message.suggestedOptions && Object.keys(message.suggestedOptions).length > 0 && (
                        <div className="mt-3 space-y-2">
                          {Object.entries(message.suggestedOptions).map(([key, options]) => (
                            <div key={key} className="space-y-1">
                              <div className="text-xs font-medium text-text-muted capitalize">
                                {key.replace(/_/g, ' ')}:
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                {options.map((option, idx) => (
                                  <button
                                    key={idx}
                                    onClick={() => handleSuggestedOption(option)}
                                    className="text-xs px-2 py-1 bg-primary/10 hover:bg-primary/20 text-primary rounded border border-primary/20 transition-colors"
                                  >
                                    {option}
                                  </button>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          }

          // Regular AI message
          return (
            <div key={message.id} className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <Bot className="w-4 h-4 text-primary" />
              </div>
              <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-surface-1 text-text">
                {message.loading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">AI is thinking...</span>
                  </div>
                ) : (
                  <ExecutionResultDisplay
                    message={message.content}
                    executionDetails={message.executionDetails}
                  />
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about this incident..."
            className="flex-1 bg-surface-1 border border-border rounded-lg px-3 py-2 text-sm text-text placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="px-4 py-2 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-bg rounded-lg transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
