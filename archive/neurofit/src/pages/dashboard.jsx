import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth, userPreferences, workouts, progress } from '../services/api'
import Header from '../components/Header'
import AIChat from '../components/AIChat'
import WorkoutList from '../components/WorkoutList'
import ProgressTracker from '../components/ProgressTracker'
import PreferencesModal from '../components/PreferencesModal'
import '../styles/dashboard.css'

function Dashboard() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [userPrefs, setUserPrefs] = useState(null)
  const [workoutList, setWorkoutList] = useState([])
  const [loading, setLoading] = useState(true)
  const [showPreferences, setShowPreferences] = useState(false)
  const [workoutCompleted, setWorkoutCompleted] = useState(0)
  const [recentActivity, setRecentActivity] = useState([])

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const currentUser = await auth.getCurrentUser()
        if (!currentUser) {
          navigate('/auth')
          return
        }

        setUser(currentUser)

        // Load user preferences
        const { data: prefs } = await userPreferences.get(currentUser.id)
        setUserPrefs(prefs)

        // If no preferences, redirect to onboarding
        if (!prefs) {
          navigate('/onboarding')
          return
        }

        // Load workouts
        const { data: workoutData, error: workoutError } = await workouts.getAll()
        if (workoutError) {
          console.error('Error loading workouts:', workoutError)
        } else {
          console.log('Loaded workouts:', workoutData?.length || 0)
          setWorkoutList(workoutData || [])
        }

        // Load recent activity for AI coach
        const { data: progressData } = await progress.getUserProgress(currentUser.id)
        setRecentActivity(progressData?.slice(0, 5) || [])

      } catch (error) {
        console.error('Error loading dashboard:', error)
      } finally {
        setLoading(false)
      }
    }

    loadDashboard()
  }, [navigate])

  const handleWorkoutComplete = async (workoutId) => {
    if (!user) return

    try {
      await progress.logWorkout(user.id, workoutId)
      setWorkoutCompleted(prev => prev + 1) // Trigger refresh
      
      // Refresh recent activity
      const { data: progressData } = await progress.getUserProgress(user.id)
      setRecentActivity(progressData?.slice(0, 5) || [])
    } catch (error) {
      console.error('Error logging workout:', error)
    }
  }

  const handlePreferencesUpdate = (updatedPrefs) => {
    setUserPrefs(updatedPrefs)
  }

  const refreshDashboard = async () => {
    if (!user) return
    const { data: prefs } = await userPreferences.get(user.id)
    setUserPrefs(prefs)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-calm-700">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="dashboard-container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl sm:text-4xl font-bold text-calm-900 mb-2">
              Welcome back{userPrefs?.name ? `, ${userPrefs.name}` : user?.email ? `, ${user.email.split('@')[0]}` : ''}
            </h1>
            <p className="text-calm-700">
              Your personalized fitness journey starts here
            </p>
          </div>
          <button
            onClick={() => setShowPreferences(true)}
            className="bg-calm-100 text-calm-700 font-medium px-4 py-2 rounded-lg hover:bg-calm-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
            aria-label="Open preferences"
          >
            Preferences
          </button>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* AI Coach Chat - Takes 2 columns on large screens */}
          <div className="lg:col-span-2">
            <AIChat userPreferences={userPrefs} workouts={workoutList} recentActivity={recentActivity} />
          </div>

          {/* Progress Tracker - Takes 1 column */}
          <div className="lg:col-span-1">
            <ProgressTracker userId={user?.id} onWorkoutComplete={workoutCompleted} />
          </div>
        </div>

        {/* Workout Library */}
        <div>
          <h2 className="text-2xl font-bold text-calm-900 mb-6">
            Workout Library
          </h2>
          <WorkoutList 
            workouts={workoutList} 
            userPreferences={userPrefs}
            onWorkoutComplete={handleWorkoutComplete}
          />
        </div>
      </main>

      <PreferencesModal
        isOpen={showPreferences}
        onClose={() => setShowPreferences(false)}
        userPrefs={userPrefs}
        onUpdate={handlePreferencesUpdate}
      />
    </div>
  )
}

export default Dashboard

