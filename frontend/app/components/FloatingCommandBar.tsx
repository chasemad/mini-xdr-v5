import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, Command, ArrowRight, Zap, Search } from 'lucide-react';
import { Button } from "@/components/ui/button";

interface FloatingCommandBarProps {
    onCommandSubmit: (command: string) => void;
    isProcessing: boolean;
}

export const FloatingCommandBar: React.FC<FloatingCommandBarProps> = ({
    onCommandSubmit,
    isProcessing
}) => {
    const [input, setInput] = useState('');
    const [isFocused, setIsFocused] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onCommandSubmit(input);
            setInput('');
        }
    };

    // Keyboard shortcut to focus (Cmd+K)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                inputRef.current?.focus();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    return (
        <div className="w-full">
            <form onSubmit={handleSubmit} className="relative group">
                {/* Glow Effect */}
                <div className={`
          absolute -inset-0.5 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-600 rounded-full opacity-20 blur transition duration-500
          ${isFocused ? 'opacity-60' : 'group-hover:opacity-40'}
        `} />

                <div className={`
          relative flex items-center bg-[#0A0A0A]/90 backdrop-blur-xl border border-white/10 rounded-full p-2 pr-2 shadow-2xl transition-all duration-300
          ${isFocused ? 'ring-1 ring-white/20 scale-[1.01]' : ''}
        `}>
                    {/* Icon */}
                    <div className="pl-4 pr-3 text-cyan-400">
                        {isProcessing ? (
                            <Sparkles className="animate-spin" size={20} />
                        ) : (
                            <Command size={20} />
                        )}
                    </div>

                    {/* Input */}
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder="Describe a workflow... (e.g. 'Block IP 1.2.3.4 if risk > 80')"
                        className="flex-1 bg-transparent border-none outline-none text-slate-100 placeholder:text-slate-500 h-10 text-base"
                        disabled={isProcessing}
                    />

                    {/* Right Actions */}
                    <div className="flex items-center gap-2">
                        {!input && (
                            <div className="hidden md:flex items-center gap-1 text-xs text-slate-600 font-mono mr-2 px-2 py-1 rounded bg-white/5 border border-white/5">
                                <span>⌘</span><span>K</span>
                            </div>
                        )}

                        <Button
                            type="submit"
                            size="icon"
                            className={`
                rounded-full w-10 h-10 transition-all duration-300
                ${input ? 'bg-cyan-500 hover:bg-cyan-400 text-black shadow-[0_0_15px_rgba(6,182,212,0.5)]' : 'bg-white/5 text-slate-500 hover:bg-white/10'}
              `}
                            disabled={!input.trim() || isProcessing}
                        >
                            <ArrowRight size={18} />
                        </Button>
                    </div>
                </div>

                {/* Suggestions / Quick Actions (Visible when focused but empty) - Opens DOWNWARDS now */}
                {isFocused && !input && (
                    <div className="absolute top-full left-0 w-full mt-4 p-2 bg-[#0A0A0A]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl animate-in slide-in-from-top-2 fade-in duration-200 z-50">
                        <div className="text-xs font-semibold text-slate-500 px-3 py-2 uppercase tracking-wider">
                            Suggested Workflows
                        </div>
                        <div className="grid gap-1">
                            {[
                                { label: "Block IP & Isolate Host", desc: "Network defense", icon: Zap },
                                { label: "Investigate Phishing", desc: "Email analysis", icon: Search },
                                { label: "Ransomware Containment", desc: "Emergency response", icon: Shield }
                            ].map((item, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onMouseDown={(e) => {
                                        e.preventDefault(); // Prevent blur
                                        setInput(item.label);
                                        onCommandSubmit(item.label); // Auto-submit
                                        setIsFocused(false);
                                    }}
                                    className="flex items-center gap-3 w-full p-3 rounded-lg hover:bg-white/5 text-left group/item transition-colors"
                                >
                                    <div className="p-2 rounded-md bg-cyan-500/10 text-cyan-400 group-hover/item:bg-cyan-500/20">
                                        <item.icon size={16} />
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium text-slate-200">{item.label}</div>
                                        <div className="text-xs text-slate-500">{item.desc}</div>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </form>
        </div>
    );
};

import { Shield } from 'lucide-react';
