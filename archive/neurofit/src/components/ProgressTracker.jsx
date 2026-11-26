import { useState, useEffect } from 'react'
import { progress } from '../services/api'
import { formatDate } from '../utils/helpers'

function ProgressTracker({ userId, onWorkoutComplete }) {
  const [stats, setStats] = useState({ totalWorkouts: 0, thisWeek: 0, streak: 0, lastCompleted: null })
  const [recentWorkouts, setRecentWorkouts] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!userId) return

    const loadProgress = async () => {
      try {
        const { stats: userStats } = await progress.getStats(userId)
        setStats(userStats || { totalWorkouts: 0, thisWeek: 0 })

        const { data: progressData } = await progress.getUserProgress(userId)
        setRecentWorkouts(progressData?.slice(0, 5) || [])
      } catch (error) {
        console.error('Error loading progress:', error)
      } finally {
        setLoading(false)
      }
    }

    loadProgress()

    // Refresh every 30 seconds to catch new completions
    const interval = setInterval(loadProgress, 30000)
    return () => clearInterval(interval)
  }, [userId, onWorkoutComplete])

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-calm-200 p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-calm-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-calm-200 rounded w-1/2"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-calm-200 p-6">
      <h2 className="text-xl font-bold text-calm-900 mb-6">Your Progress</h2>

      {/* Stats */}
      <div className="space-y-4 mb-6">
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-calm-600">Total Workouts</span>
            <span className="text-2xl font-bold text-primary-600">{stats.totalWorkouts}</span>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-calm-600">Current Streak</span>
            <span className="text-xl font-semibold text-primary-600">{stats.streak} {stats.streak === 1 ? 'day' : 'days'}</span>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-calm-600">This Week</span>
            <span className="text-xl font-semibold text-calm-900">{stats.thisWeek}</span>
          </div>
          <div className="w-full bg-calm-200 rounded-full h-2">
            <div
              className="progress-bar bg-primary-600 h-2 rounded-full"
              style={{ width: `${Math.min((stats.thisWeek / 7) * 100, 100)}%` }}
              role="progressbar"
              aria-valuenow={stats.thisWeek}
              aria-valuemin="0"
              aria-valuemax="7"
            ></div>
          </div>
        </div>

        {stats.lastCompleted && (
          <div>
            <div className="text-xs text-calm-500">
              Last completed: {formatDate(stats.lastCompleted)}
            </div>
          </div>
        )}
      </div>

      {/* Recent Workouts */}
      <div>
        <h3 className="text-sm font-semibold text-calm-700 mb-3">Recent Completions</h3>
        {recentWorkouts.length === 0 ? (
          <p className="text-sm text-calm-600">No workouts completed yet. Start your first workout!</p>
        ) : (
          <ul className="space-y-2">
            {recentWorkouts.map((item) => (
              <li key={item.id} className="text-sm text-calm-700">
                <div className="flex justify-between items-start">
                  <span className="font-medium">
                    {item.workouts?.title || 'Workout'}
                  </span>
                  <span className="text-calm-500 text-xs ml-2">
                    {formatDate(item.completed_at)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

export default ProgressTracker

