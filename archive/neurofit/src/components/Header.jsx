import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { auth } from '../services/api'

function Header() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkUser = async () => {
      const currentUser = await auth.getCurrentUser()
      setUser(currentUser)
      setLoading(false)
    }
    checkUser()

    // Listen for auth changes
    const { data: { subscription } } = auth.supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => subscription.unsubscribe()
  }, [])

  const handleLogout = async () => {
    await auth.signOut()
    navigate('/')
  }

  return (
    <header className="bg-white shadow-sm" role="banner">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4" aria-label="Main navigation">
        <div className="flex items-center justify-between">
          <Link 
            to="/" 
            className="text-2xl font-bold text-calm-800 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md transition-colors"
          >
            NeuroFit
          </Link>
          
          <div className="flex items-center gap-4">
            {loading ? (
              <span className="text-calm-600">Loading...</span>
            ) : user ? (
              <>
                <Link
                  to="/dashboard"
                  className="text-calm-700 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md px-3 py-2 transition-colors"
                >
                  Dashboard
                </Link>
                <button
                  onClick={handleLogout}
                  className="text-calm-700 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md px-3 py-2 transition-colors"
                  aria-label="Sign out"
                >
                  Sign Out
                </button>
              </>
            ) : (
              <Link
                to="/auth"
                className="bg-primary-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </nav>
    </header>
  )
}

export default Header

