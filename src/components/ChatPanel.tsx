import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, ThumbsUp, ThumbsDown, Mic, Copy, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { ChatMessage, initialMessages } from '@/lib/dummyData';
import { sendAgentMessage } from '@/lib/agentApi';

export const ChatPanel = () => {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [actionOptions, setActionOptions] = useState<string[]>([]);
  const [canForceConflict, setCanForceConflict] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [speechMessage, setSpeechMessage] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [feedbacks, setFeedbacks] = useState<Record<string, 'up' | 'down'>>({});
  const [copied, setCopied] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | undefined>(undefined);
  const endRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  const isTrustedLocalOrigin = () => {
    const host = window.location.hostname.toLowerCase();
    if (host === 'localhost' || host === '::1' || host === '[::1]') {
      return true;
    }
    if (host.endsWith('.localhost')) {
      return true;
    }
    if (/^127(?:\.\d{1,3}){3}$/.test(host)) {
      return true;
    }
    return false;
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const isLocalhost = isTrustedLocalOrigin();
    if (!window.isSecureContext && !isLocalhost) {
      setSpeechSupported(false);
      setSpeechMessage('Voice input requires HTTPS or localhost. Open http://localhost:8080 in Chrome or Edge.');
      return;
    }

    const speechApi = (window as Window & { SpeechRecognition?: any; webkitSpeechRecognition?: any }).SpeechRecognition
      || (window as Window & { SpeechRecognition?: any; webkitSpeechRecognition?: any }).webkitSpeechRecognition;

    if (!speechApi) {
      setSpeechSupported(false);
      setSpeechMessage('Speech recognition is not supported in this browser/webview. Use Chrome or Edge at http://localhost:8080.');
      return;
    }

    setSpeechSupported(true);
    setSpeechMessage(null);
    const recognition = new speechApi();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setSpeechMessage('Listening...');
    };

    recognition.onresult = (event: any) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        transcript += event.results[i][0].transcript;
      }
      setInput(transcript.trim());
      setSpeechMessage(null);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onerror = (event: any) => {
      setIsListening(false);
      const code = String(event?.error || '').toLowerCase();
      if (code === 'not-allowed' || code === 'service-not-allowed') {
        setSpeechMessage('Microphone permission denied. Allow mic access in browser site settings.');
      } else if (code === 'audio-capture') {
        setSpeechMessage('No microphone was detected on this device.');
      } else if (code === 'network') {
        setSpeechMessage('Voice recognition network error. Please try again.');
      } else if (code === 'no-speech') {
        setSpeechMessage('No speech detected. Try speaking again.');
      } else {
        setSpeechMessage('Voice input failed to start. Please try again.');
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const toggleVoiceInput = async () => {
    if (!recognitionRef.current) {
      const isLocalhost = isTrustedLocalOrigin();
      if (!window.isSecureContext && !isLocalhost) {
        setSpeechMessage('Voice input is blocked on non-secure origins. Open http://localhost:8080 (or use HTTPS).');
      } else {
        setSpeechMessage('Speech recognition API is unavailable in this browser/webview. Open http://localhost:8080 in Chrome or Edge.');
      }
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    try {
      if (navigator.mediaDevices?.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      }
      recognitionRef.current.start();
    } catch {
      setIsListening(false);
      setSpeechMessage('Microphone permission denied. Please allow access and try again.');
    }
  };

  const send = async (text?: string) => {
    const message = text || input;
    if (!message.trim()) return;

    const userMsg: ChatMessage = { id: Date.now().toString(), role: 'user', content: message, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const result = await sendAgentMessage({
        message: userMsg.content,
        threadId,
      });

      setThreadId(result.thread_id);

      const response: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.reply,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, response]);
      setActionOptions(Array.isArray(result.action_options) ? result.action_options : []);
      setCanForceConflict(result.booking_status === 'conflict');
      window.dispatchEvent(new Event('booking-data-updated'));
    } catch (error) {
      const fallbackText =
        error instanceof Error
          ? `I could not reach the booking backend. ${error.message}`
          : 'I could not reach the booking backend.';

      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `${fallbackText}\n\nPlease ensure the Python API is running on http://127.0.0.1:8000.`,
          timestamp: new Date(),
        },
      ]);
      setActionOptions([]);
      setCanForceConflict(false);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              <div className={`glass-button rounded-full p-2 h-fit shrink-0 ${msg.role === 'assistant' ? 'text-primary' : 'text-accent'}`}>
                {msg.role === 'assistant' ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
              </div>
              <div className={`glass-card p-4 max-w-[80%] ${msg.role === 'user' ? 'rounded-2xl rounded-tr-md' : 'rounded-2xl rounded-tl-md'}`}>
                <div className="prose prose-sm dark:prose-invert max-w-none text-foreground text-sm leading-relaxed">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
                <span className="text-[10px] text-muted-foreground mt-2 block">
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                {msg.role === 'assistant' && msg.id !== '1' && (
                  <div className="flex items-center gap-1 mt-2 pt-2 border-t border-border/20">
                    <button
                      onClick={() => setFeedbacks(prev => ({ ...prev, [msg.id]: 'up' }))}
                      className={`glass-button rounded-lg p-1.5 transition-all ${feedbacks[msg.id] === 'up' ? 'text-primary ring-1 ring-primary/30' : 'text-muted-foreground hover:text-primary'}`}
                    >
                      <ThumbsUp className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => setFeedbacks(prev => ({ ...prev, [msg.id]: 'down' }))}
                      className={`glass-button rounded-lg p-1.5 transition-all ${feedbacks[msg.id] === 'down' ? 'text-destructive ring-1 ring-destructive/30' : 'text-muted-foreground hover:text-destructive'}`}
                    >
                      <ThumbsDown className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(msg.content);
                        setCopied(msg.id);
                        setTimeout(() => setCopied(null), 1500);
                      }}
                      className="glass-button rounded-lg p-1.5 text-muted-foreground hover:text-foreground transition-all ml-auto"
                    >
                      {copied === msg.id ? <Check className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isTyping && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
            <div className="glass-button rounded-full p-2 text-primary"><Bot className="h-4 w-4" /></div>
            <div className="glass-card p-4 rounded-2xl rounded-tl-md">
              <div className="flex gap-1.5">
                {[0, 1, 2].map(i => (
                  <motion.div key={i} className="w-2 h-2 rounded-full bg-primary/60"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
        <div ref={endRef} />
      </div>

      <div className="p-4 border-t border-border/50">
        <div className="flex flex-wrap items-center gap-2 mb-2">
          <button
            onClick={() => void send('edit my booking')}
            className="glass-button rounded-full px-3 py-1 text-[10px] text-muted-foreground hover:text-foreground"
          >
            Edit booking
          </button>
          <button
            onClick={() => void send('reschedule my meeting')}
            className="glass-button rounded-full px-3 py-1 text-[10px] text-muted-foreground hover:text-foreground"
          >
            Reschedule
          </button>
        </div>

        {actionOptions.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 mb-2">
            {actionOptions.map((_, index) => (
              <button
                key={`action-option-${index}`}
                onClick={() => void send(`option ${index + 1}`)}
                className="glass-button rounded-full px-3 py-1 text-[10px] text-primary"
              >
                Option {index + 1}
              </button>
            ))}
            {canForceConflict && (
              <button
                onClick={() => void send('go with conflict')}
                className="glass-button rounded-full px-3 py-1 text-[10px] text-destructive"
              >
                Go with conflict
              </button>
            )}
          </div>
        )}

        <div className="glass-input flex items-center gap-2 rounded-2xl px-4 py-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="Type a message... e.g. 'Book a meeting tomorrow at 3 PM'"
            className="flex-1 bg-transparent outline-none text-sm text-foreground placeholder:text-muted-foreground"
          />
          <motion.button
            whileTap={{ scale: 0.85 }}
            onClick={() => send()}
            disabled={!input.trim()}
            className="glass-button rounded-full p-2.5 text-primary disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </motion.button>
          <motion.button
            whileTap={{ scale: 0.85 }}
            onClick={() => void toggleVoiceInput()}
            className={`glass-button rounded-full p-2.5 ${isListening ? 'text-primary ring-1 ring-primary/40' : 'text-muted-foreground hover:text-accent'}`}
            title={speechSupported ? (isListening ? 'Stop voice input' : 'Start voice input') : 'Voice input unavailable: click for details'}
          >
            <Mic className="h-4 w-4" />
          </motion.button>
        </div>
        {speechMessage && (
          <p className="mt-2 text-[11px] text-muted-foreground">{speechMessage}</p>
        )}
      </div>
    </div>
  );
};
