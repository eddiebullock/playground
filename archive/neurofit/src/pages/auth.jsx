import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth } from '../services/api'
import { validateEmail, validatePassword } from '../utils/helpers'
import Header from '../components/Header'

function Auth() {
  const navigate = useNavigate()
  const [isSignUp, setIsSignUp] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    if (!validateEmail(email)) {
      setError('Please enter a valid email address')
      setLoading(false)
      return
    }

    if (!validatePassword(password)) {
      setError('Password must be at least 6 characters')
      setLoading(false)
      return
    }

    try {
      if (isSignUp) {
        const { error: signUpError } = await auth.signUp(email, password)
        if (signUpError) {
          setError(signUpError.message || 'Failed to sign up. Please try again.')
        } else {
          // After signup, redirect to onboarding
          navigate('/onboarding')
        }
      } else {
        const { error: signInError } = await auth.signIn(email, password)
        if (signInError) {
          setError(signInError.message || 'Failed to sign in. Please check your credentials.')
        } else {
          // After signin, check if user has preferences
          const { userPreferences } = await import('../services/api')
          const user = await auth.getCurrentUser()
          if (user) {
            const { data: prefs } = await userPreferences.get(user.id)
            if (prefs) {
              navigate('/dashboard')
            } else {
              navigate('/onboarding')
            }
          }
        }
      }
    } catch (err) {
      setError('Something went wrong. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      <div className="max-w-md mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-3xl font-bold text-calm-900 mb-2 text-center">
          {isSignUp ? 'Create Account' : 'Sign In'}
        </h1>
        <p className="text-calm-700 mb-8 text-center">
          {isSignUp 
            ? 'Join NeuroFit and start your personalized fitness journey'
            : 'Welcome back to NeuroFit'
          }
        </p>

        {error && (
          <div 
            className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
            role="alert"
          >
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label 
              htmlFor="email"
              className="block text-sm font-medium text-calm-700 mb-2"
            >
              Email
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="your@email.com"
            />
          </div>

          <div>
            <label 
              htmlFor="password"
              className="block text-sm font-medium text-calm-700 mb-2"
            >
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="w-full px-4 py-3 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="At least 6 characters"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-primary-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Please wait...' : (isSignUp ? 'Sign Up' : 'Sign In')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp)
              setError('')
            }}
            className="text-primary-600 hover:text-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md"
          >
            {isSignUp 
              ? 'Already have an account? Sign in'
              : "Don't have an account? Sign up"
            }
          </button>
        </div>
      </div>
    </div>
  )
}

export default Auth

