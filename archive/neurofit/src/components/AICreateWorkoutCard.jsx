import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function AICreateWorkoutCard({ onView }) {
  const navigate = useNavigate()

  const handleClick = () => {
    // Scroll to AI chat or open a modal
    const aiChatElement = document.querySelector('[data-ai-chat]')
    if (aiChatElement) {
      aiChatElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      // Focus on the chat input
      setTimeout(() => {
        const chatInput = aiChatElement.querySelector('input[type="text"]')
        if (chatInput) {
          chatInput.focus()
          chatInput.value = 'Create a custom workout for me'
          chatInput.dispatchEvent(new Event('input', { bubbles: true }))
        }
      }, 500)
    }
  }

  return (
    <div 
      className="workout-card bg-gradient-to-br from-primary-50 to-primary-100 rounded-lg shadow-sm border-2 border-primary-200 p-6 hover:shadow-md cursor-pointer transition-all relative overflow-hidden"
      onClick={handleClick}
    >
      <div className="absolute top-0 right-0 w-32 h-32 bg-primary-200 opacity-20 rounded-full -mr-16 -mt-16"></div>
      <div className="relative">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-6 h-6 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <h3 className="text-xl font-bold text-primary-900">Create Custom Workout</h3>
        </div>
        
        <p className="text-primary-800 mb-4 leading-relaxed">
          Use our AI coach to design a personalized workout tailored to your specific needs, preferences, and goals.
        </p>

        <div className="flex flex-wrap gap-2 mb-4">
          <span className="px-2 py-1 bg-primary-200 text-primary-900 text-xs rounded font-medium">
            AI-Powered
          </span>
          <span className="px-2 py-1 bg-primary-200 text-primary-900 text-xs rounded font-medium">
            Personalized
          </span>
          <span className="px-2 py-1 bg-primary-200 text-primary-900 text-xs rounded font-medium">
            Custom
          </span>
        </div>

        <div className="text-sm text-primary-700 font-medium">
          Click to start →
        </div>
      </div>
    </div>
  )
}

export default AICreateWorkoutCard

