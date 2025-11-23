"use client";

import React, { useState, useEffect, useRef } from "react";
import { Bot, Send, X, Sparkles, Loader2, User, Terminal, HelpCircle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { agentOrchestrate, confirmAgentAction } from "@/app/lib/api";
import { cn } from "@/lib/utils";
import { ConfirmationPrompt } from "@/components/ConfirmationPrompt";
import { ExecutionResultDisplay } from "@/components/ExecutionResultDisplay";

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

interface CopilotSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  selectedIncidentId?: number;
  incidentData?: any;
}

export function CopilotSidebar({
  isOpen,
  onClose,
  selectedIncidentId,
  incidentData
}: CopilotSidebarProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      type: 'ai',
      content: "Welcome to the SOC Command Center. I'm your AI analyst assistant. I can help you investigate incidents, analyze threats, and coordinate response actions. What would you like to explore?",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [pendingActionId, setPendingActionId] = useState<string | undefined>();
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add temporary loading message
    const loadingId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: loadingId,
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    }]);

    try {
      const response = await agentOrchestrate(userMessage.content, selectedIncidentId, {
        incident_data: incidentData,
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
        // Clear pending action if completed
        setPendingActionId(undefined);
      }

      // Handle different response types
      const responseType = response.response_type || 'answer';

      let newMessage: ChatMessage;

      if (responseType === 'confirmation_required') {
        // Confirmation prompt
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
        // Follow-up questions
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
        // Regular answer or execution result
        newMessage = {
          id: (Date.now() + 2).toString(),
          type: 'ai',
          content: response.message || response.analysis || "I've processed your request.",
          timestamp: new Date(),
          executionDetails: response.execution_details
        };
      }

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg =>
        msg.id === loadingId ? newMessage : msg
      ));

    } catch (error) {
      console.error('AI response failed:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === loadingId
          ? {
              id: (Date.now() + 2).toString(),
              type: 'ai',
              content: "I apologize, but I encountered an error processing your request. Please try again.",
              timestamp: new Date()
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfirmation = async (approved: boolean, msgPendingActionId?: string) => {
    if (!msgPendingActionId) return;

    setIsLoading(true);

    // Add loading message
    const loadingId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: loadingId,
      type: 'ai',
      content: '',
      timestamp: new Date(),
      loading: true
    }]);

    try {
      const response = await confirmAgentAction(
        msgPendingActionId,
        approved,
        selectedIncidentId,
        { incident_data: incidentData }
      );

      // Clear pending action
      setPendingActionId(undefined);

      const newMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'ai',
        content: response.message || (approved ? "Action approved and executed!" : "Action cancelled."),
        timestamp: new Date()
      };

      setMessages(prev => prev.map(msg =>
        msg.id === loadingId ? newMessage : msg
      ));

    } catch (error) {
      console.error('Confirmation failed:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === loadingId
          ? {
              id: (Date.now() + 2).toString(),
              type: 'ai',
              content: "Failed to process confirmation. Please try again.",
              timestamp: new Date()
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedOption = (option: string) => {
    setInput(option);
  };

  const handleClearChat = () => {
    // Reset to initial welcome message
    setMessages([{
      id: 'welcome',
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
    <div
      className={cn(
        "fixed inset-y-0 right-0 z-50 w-96 bg-background border-l border-border shadow-2xl transform transition-transform duration-300 ease-in-out flex flex-col",
        isOpen ? "translate-x-0" : "translate-x-full"
      )}
    >
      {/* Header */}
      <div className="h-14 border-b border-border flex items-center justify-between px-4 bg-muted/30">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-primary/10 rounded-md">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Security Copilot</h3>
            <p className="text-[10px] text-muted-foreground flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              Online
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 1 && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearChat}
              className="h-8 w-8"
              title="New conversation"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((msg) => {
            // User message
            if (msg.type === 'user') {
              return (
                <div key={msg.id} className="flex gap-3 text-sm justify-end">
                  <div className="rounded-lg px-3 py-2 max-w-[85%] bg-primary text-primary-foreground">
                    <div className="whitespace-pre-wrap leading-relaxed">
                      {msg.content}
                    </div>
                    <div className="text-[10px] mt-1 opacity-50 text-primary-foreground">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0 border border-border">
                    <User className="w-4 h-4 text-muted-foreground" />
                  </div>
                </div>
              );
            }

            // Confirmation message
            if (msg.type === 'confirmation') {
              return (
                <div key={msg.id} className="space-y-2">
                  <div className="flex gap-3 text-sm justify-start">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 border border-primary/20">
                      <Sparkles className="w-4 h-4 text-primary" />
                    </div>
                    <div className="flex-1 max-w-[90%]">
                      <ConfirmationPrompt
                        actionPlan={msg.actionPlan}
                        message={msg.content}
                        onApprove={() => handleConfirmation(true, msg.pendingActionId)}
                        onReject={() => handleConfirmation(false, msg.pendingActionId)}
                        isLoading={isLoading}
                      />
                    </div>
                  </div>
                </div>
              );
            }

            // Follow-up questions message
            if (msg.type === 'follow_up') {
              return (
                <div key={msg.id} className="space-y-2">
                  <div className="flex gap-3 text-sm justify-start">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 border border-primary/20">
                      <HelpCircle className="w-4 h-4 text-primary" />
                    </div>
                    <div className="flex-1 max-w-[85%]">
                      <div className="rounded-lg px-3 py-2 bg-muted border border-border">
                        <div className="whitespace-pre-wrap leading-relaxed">
                          {msg.content}
                        </div>

                        {msg.followUpQuestions && msg.followUpQuestions.length > 0 && (
                          <div className="mt-3 space-y-2">
                            {msg.followUpQuestions.map((question, idx) => (
                              <div key={idx} className="text-xs text-muted-foreground flex items-start gap-1">
                                <span className="mt-0.5">•</span>
                                <span>{question}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {msg.suggestedOptions && Object.keys(msg.suggestedOptions).length > 0 && (
                          <div className="mt-3 space-y-2">
                            {Object.entries(msg.suggestedOptions).map(([key, options]) => (
                              <div key={key} className="space-y-1">
                                <div className="text-xs font-medium text-muted-foreground capitalize">
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

                        <div className="text-[10px] mt-2 opacity-50 text-muted-foreground">
                          {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            }

            // Regular AI message
            return (
              <div key={msg.id} className="flex gap-3 text-sm justify-start">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 border border-primary/20">
                  <Sparkles className="w-4 h-4 text-primary" />
                </div>
                <div className="rounded-lg px-3 py-2 max-w-[85%] bg-muted border border-border">
                  {msg.loading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span className="text-xs opacity-70">Analyzing...</span>
                    </div>
                  ) : (
                    <ExecutionResultDisplay
                      message={msg.content}
                      executionDetails={msg.executionDetails}
                    />
                  )}
                  <div className="text-[10px] mt-1 opacity-50 text-muted-foreground">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t border-border bg-background">
        {selectedIncidentId && (
           <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded border border-border/50">
             <Terminal className="w-3 h-3" />
             Context: Incident #{selectedIncidentId}
           </div>
        )}
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Ask about threats or take action..."
            className="flex-1"
            disabled={isLoading}
            autoComplete="off"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            size="icon"
            className="shrink-0"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground mt-2 text-center">
          AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}
