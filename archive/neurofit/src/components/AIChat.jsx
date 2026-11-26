import { useState, useRef, useEffect } from 'react'
import OpenAI from 'openai'
import {
  getWorkoutRecommendationPrompt,
  getMotivationPrompt,
  getSensoryOverloadPrompt,
  getExecutiveFunctionPrompt,
} from '../utils/prompts'

// Replace with your OpenAI API key
// Get it from: https://platform.openai.com/api-keys
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || 'YOUR_OPENAI_API_KEY'

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  dangerouslyAllowBrowser: true, // Note: In production, use a backend proxy
})

function AIChat({ userPreferences, workouts, recentActivity }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm your NeuroFit AI coach. I'm here to help you with workouts, motivation, and any questions you have. How can I help you today?",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const detectIntent = (message) => {
    const lowerMessage = message.toLowerCase()
    
    if (lowerMessage.includes('motivation') || lowerMessage.includes('tired') || lowerMessage.includes('energy')) {
      return 'motivation'
    }
    if (lowerMessage.includes('sensory') || lowerMessage.includes('overwhelmed') || lowerMessage.includes('too much')) {
      return 'sensory'
    }
    if (lowerMessage.includes('start') || lowerMessage.includes('begin') || lowerMessage.includes('how do i')) {
      return 'executive'
    }
    if (lowerMessage.includes('workout') || lowerMessage.includes('exercise') || lowerMessage.includes('routine')) {
      return 'workout'
    }
    return 'general'
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const intent = detectIntent(userMessage)
      let prompt

      switch (intent) {
        case 'motivation':
          prompt = getMotivationPrompt(userMessage, userPreferences, recentActivity)
          break
        case 'sensory':
          prompt = getSensoryOverloadPrompt(userMessage, userPreferences, recentActivity)
          break
        case 'executive':
          prompt = getExecutiveFunctionPrompt(userMessage, userPreferences, recentActivity)
          break
        case 'workout':
          prompt = getWorkoutRecommendationPrompt(userMessage, userPreferences, workouts, recentActivity)
          break
        default:
          prompt = getWorkoutRecommendationPrompt(userMessage, userPreferences, workouts, recentActivity)
      }

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini', // Using gpt-4o-mini - the cheapest model. DO NOT CHANGE.
        messages: [
          { role: 'system', content: prompt },
          { role: 'user', content: userMessage },
        ],
        temperature: 0.7,
        max_tokens: 500,
      })

      const assistantMessage = completion.choices[0]?.message?.content || 
        "I'm sorry, I'm having trouble responding right now. Please try again."

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: assistantMessage,
      }])
    } catch (error) {
      console.error('Error calling OpenAI:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "I'm sorry, I'm having technical difficulties. Please check that your OpenAI API key is configured correctly.",
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-calm-200 flex flex-col h-[600px]" data-ai-chat>
      <div className="p-4 border-b border-calm-200">
        <h2 className="text-xl font-bold text-calm-900">AI Coach</h2>
        <p className="text-sm text-calm-600">Your supportive fitness companion</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 chat-message ${
                msg.role === 'user'
                  ? 'bg-primary-600 text-white'
                  : 'bg-calm-100 text-calm-900'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-calm-100 rounded-lg p-3">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-calm-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-calm-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-calm-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSend} className="p-4 border-t border-calm-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything about fitness..."
            className="flex-1 px-4 py-2 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={loading}
            aria-label="Chat input"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}

export default AIChat

