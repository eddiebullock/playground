'use client';
import React, { useState, useRef, useEffect } from 'react';
import PaperCitation from './PaperCitation';
import CrisisAlert from './CrisisAlert';
import ReactMarkdown from 'react-markdown';

function cleanMarkdown(text) {
  // Remove excessive *** and trim leading/trailing whitespace
  return text.replace(/\*\*\*+/g, '').replace(/\n{3,}/g, '\n\n').trim();
}

export default function MentalHealthChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [typing, setTyping] = useState(false);
  const [crisis, setCrisis] = useState(null);
  const [searchTerms, setSearchTerms] = useState([]);
  const [inputAtBottom, setInputAtBottom] = useState(false);
  const [animatedText, setAnimatedText] = useState('');
  const [isAnimating, setIsAnimating] = useState(false);
  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const animationIntervalRef = useRef(null);
  const fetchControllerRef = useRef(null);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages, typing, inputAtBottom, animatedText]);

  // Transition input bar to bottom after first message
  useEffect(() => {
    if (messages.length > 0 && !inputAtBottom) {
      setTimeout(() => setInputAtBottom(true), 150);
    }
  }, [messages, inputAtBottom]);

  // Typewriter animation for assistant responses
  useEffect(() => {
    if (messages.length === 0) return;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg.role === 'assistant' && lastMsg.content && !lastMsg.animated && !isAnimating) {
      setIsAnimating(true);
      let i = 0;
      const cleanText = cleanMarkdown(lastMsg.content);
      setAnimatedText('');
      if (animationIntervalRef.current) clearInterval(animationIntervalRef.current);
      animationIntervalRef.current = setInterval(() => {
        setAnimatedText(cleanText.slice(0, i + 1));
        i++;
        if (i >= cleanText.length) {
          clearInterval(animationIntervalRef.current);
          animationIntervalRef.current = null;
          setIsAnimating(false);
          // Mark this message as animated so it doesn't re-animate
          setMessages(msgs => {
            const newMsgs = [...msgs];
            newMsgs[newMsgs.length - 1] = { ...lastMsg, animated: true };
            return newMsgs;
          });
        }
      }, 12); // Speed of typing
      return () => {
        if (animationIntervalRef.current) clearInterval(animationIntervalRef.current);
      };
    }
  }, [messages]);

  // Stop/cancel logic
  const handleStop = () => {
    // Stop animation
    if (animationIntervalRef.current) {
      clearInterval(animationIntervalRef.current);
      animationIntervalRef.current = null;
      setIsAnimating(false);
      setMessages(msgs => {
        if (msgs.length === 0) return msgs;
        const last = msgs[msgs.length - 1];
        if (last.role === 'assistant' && !last.animated) {
          return [...msgs.slice(0, -1), { ...last, animated: true }];
        }
        return msgs;
      });
    }
    // Abort fetch
    if (fetchControllerRef.current) {
      fetchControllerRef.current.abort();
      fetchControllerRef.current = null;
      setLoading(false);
      setTyping(false);
    }
  };

  // When a new user message is sent, mark the last assistant message as animated and stop animation
  const sendMessage = async () => {
    if (!input.trim()) return;
    // Mark last assistant message as animated and stop animation
    setMessages(msgs => {
      if (msgs.length === 0) return [...msgs, { role: 'user', content: input }];
      const last = msgs[msgs.length - 1];
      if (last.role === 'assistant' && !last.animated) {
        return [...msgs.slice(0, -1), { ...last, animated: true }, { role: 'user', content: input }];
      }
      return [...msgs, { role: 'user', content: input }];
    });
    if (animationIntervalRef.current) {
      clearInterval(animationIntervalRef.current);
      animationIntervalRef.current = null;
      setIsAnimating(false);
    }
    setLoading(true);
    setTyping(true);
    setInput('');
    // Setup fetch abort controller
    const controller = new AbortController();
    fetchControllerRef.current = controller;
    try {
      const res = await fetch('/api/mental-health-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
        signal: controller.signal
      });
      fetchControllerRef.current = null;
      const data = await res.json();
      setSearchTerms(data.searchTermsUsed || []);
      if (data.crisisDetected) {
        setCrisis(data.crisisResources);
        setMessages(msgs => [...msgs, { role: 'assistant', content: data.response }]);
      } else {
        setCrisis(null);
        setMessages(msgs => [
          ...msgs,
          {
            role: 'assistant',
            content: data.response,
            papers: data.papers || []
          }
        ]);
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        setMessages(msgs => [...msgs, { role: 'assistant', content: 'Response stopped by user.' }]);
      } else {
        setMessages(msgs => [...msgs, { role: 'assistant', content: 'Sorry, something went wrong.' }]);
      }
    }
    setLoading(false);
    setTyping(false);
  };

  return (
    <div className="flex flex-col min-h-screen w-full bg-gradient-to-b from-gray-50 to-gray-200 dark:from-gray-900 dark:to-gray-800">
      {/* If no messages, center the input bar vertically */}
      {messages.length === 0 && !inputAtBottom ? (
        <div className="flex-1 flex flex-col items-center justify-center w-full transition-all duration-500 ease-in-out">
          <form
            className="w-full max-w-xl flex items-center gap-2 px-4 py-6 bg-white/95 dark:bg-gray-900/95 border border-gray-200 dark:border-gray-700 rounded-xl shadow-lg transition-all duration-500 ease-in-out"
            style={{ zIndex: 10 }}
            onSubmit={e => { e.preventDefault(); sendMessage(); }}
          >
            <input
              ref={inputRef}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-3 text-base bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your mental health question..."
              disabled={loading}
              autoComplete="off"
            />
            <button
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-5 py-3 rounded-lg transition-colors disabled:opacity-50"
              type="submit"
              disabled={loading || !input.trim()}
            >
              Send
            </button>
            { (loading || typing || isAnimating) && (
              <button
                type="button"
                className="ml-2 px-3 py-2 bg-red-100 text-red-700 rounded-lg text-xs font-semibold border border-red-300 hover:bg-red-200 transition-colors"
                onClick={handleStop}
              >
                Stop
              </button>
            )}
          </form>
        </div>
      ) : (
        <>
          {/* Chat area */}
          <main ref={chatRef} className="flex-1 overflow-y-auto px-0 py-4 sm:py-8 w-full transition-all duration-500 ease-in-out" style={{ paddingBottom: inputAtBottom ? 96 : 0 }}>
            <div className="max-w-2xl mx-auto w-full h-full flex flex-col">
              {crisis && <CrisisAlert resources={crisis} />}
              <div className="flex flex-col gap-4 w-full">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`rounded-xl px-4 py-3 shadow-sm max-w-[90vw] sm:max-w-[70%] whitespace-pre-line break-words ${msg.role === 'user' ? 'bg-indigo-600 text-white dark:bg-indigo-500' : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700'}`}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs font-semibold ${msg.role === 'user' ? 'text-indigo-200' : 'text-indigo-600 dark:text-indigo-300'}`}>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                      </div>
                      <div className="text-base leading-relaxed">
                        {msg.role === 'assistant' && i === messages.length - 1 && !msg.animated ? (
                          <ReactMarkdown>{animatedText}</ReactMarkdown>
                        ) : msg.role === 'assistant' ? (
                          <ReactMarkdown>{cleanMarkdown(msg.content)}</ReactMarkdown>
                        ) : (
                          msg.content
                        )}
                      </div>
                      {msg.papers && msg.papers.length > 0 && (
                        <div className="mt-3">
                          <details className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded p-2">
                            <summary className="cursor-pointer text-sm text-indigo-700 dark:text-indigo-300 font-semibold">Show research citations</summary>
                            {msg.papers.map((p, j) => <PaperCitation key={j} paper={p} />)}
                          </details>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {typing && (
                  <div className="flex w-full justify-start">
                    <div className="rounded-xl px-4 py-3 shadow-sm max-w-[70%] bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700 animate-pulse">
                      <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-300 mb-1 block">Assistant</span>
                      <span className="text-base">Assistant is typing...</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </main>
          {/* Input bar fixed to bottom */}
          <form
            className={`w-full flex items-center gap-2 px-4 py-3 bg-white/95 dark:bg-gray-900/95 border-t border-gray-200 dark:border-gray-700 fixed bottom-0 left-0 transition-all duration-500 ease-in-out ${inputAtBottom ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10 pointer-events-none'}`}
            style={{ zIndex: 10 }}
            onSubmit={e => { e.preventDefault(); sendMessage(); }}
          >
            <input
              ref={inputRef}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-3 text-base bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your mental health question..."
              disabled={loading}
              autoComplete="off"
            />
            <button
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-5 py-3 rounded-lg transition-colors disabled:opacity-50"
              type="submit"
              disabled={loading || !input.trim()}
            >
              Send
            </button>
            { (loading || typing || isAnimating) && (
              <button
                type="button"
                className="ml-2 px-3 py-2 bg-red-100 text-red-700 rounded-lg text-xs font-semibold border border-red-300 hover:bg-red-200 transition-colors"
                onClick={handleStop}
              >
                Stop
              </button>
            )}
          </form>
        </>
      )}
    </div>
  );
} 