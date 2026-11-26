import { useState, useMemo } from 'react'
import WorkoutCard from './WorkoutCard'
import WorkoutDetailModal from './WorkoutDetailModal'
import AICreateWorkoutCard from './AICreateWorkoutCard'
import { filterWorkoutsByPreferences } from '../utils/helpers'

function WorkoutList({ workouts, userPreferences, onWorkoutComplete }) {
  const [selectedWorkout, setSelectedWorkout] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedTag, setSelectedTag] = useState('')
  const [showAll, setShowAll] = useState(false) // Temporary debug option

  // Get all unique tags
  const allTags = useMemo(() => {
    const tags = new Set()
    workouts.forEach(workout => {
      if (workout.tags && Array.isArray(workout.tags)) {
        workout.tags.forEach(tag => tags.add(tag))
      }
    })
    return Array.from(tags)
  }, [workouts])

  // Filter workouts
  const filteredWorkouts = useMemo(() => {
    console.log('Filtering workouts. Total:', workouts?.length || 0)
    console.log('User preferences:', userPreferences)
    
    // If showAll is enabled, skip filtering
    if (showAll) {
      console.log('Show All enabled - skipping preference filter')
      return workouts || []
    }
    
    let filtered = workouts || []
    
    // Only filter by preferences if preferences exist
    if (userPreferences && workouts) {
      filtered = filterWorkoutsByPreferences(workouts, userPreferences)
      console.log('After preference filter:', filtered.length)
    }

    if (searchTerm) {
      filtered = filtered.filter(workout =>
        workout.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        workout.description?.toLowerCase().includes(searchTerm.toLowerCase())
      )
      console.log('After search filter:', filtered.length)
    }

    if (selectedTag) {
      filtered = filtered.filter(workout =>
        workout.tags && Array.isArray(workout.tags) && workout.tags.some(tag => 
          tag.toLowerCase() === selectedTag.toLowerCase()
        )
      )
      console.log('After tag filter:', filtered.length)
    }

    console.log('Final filtered workouts:', filtered.length)
    return filtered
  }, [workouts, userPreferences, searchTerm, selectedTag, showAll])

  if (!workouts || workouts.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-calm-200 p-8 text-center">
        <p className="text-calm-700 mb-2">No workouts available yet. Check back soon!</p>
        <p className="text-sm text-calm-500">Debug: workouts prop is {workouts ? `${workouts.length} items` : 'null/undefined'}</p>
      </div>
    )
  }

  return (
    <div>
      {/* Search and Filter */}
      <div className="mb-6 space-y-4 sm:flex sm:space-y-0 sm:gap-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search workouts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            aria-label="Search workouts"
          />
        </div>
        {allTags.length > 0 && (
          <select
            value={selectedTag}
            onChange={(e) => setSelectedTag(e.target.value)}
            className="px-4 py-2 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white"
            aria-label="Filter by tag"
          >
            <option value="">All Tags</option>
            {allTags.map(tag => (
              <option key={tag} value={tag}>{tag}</option>
            ))}
          </select>
        )}
        {/* Debug: Show All button */}
        <button
          onClick={() => setShowAll(!showAll)}
          className={`px-4 py-2 rounded-lg border text-sm ${
            showAll 
              ? 'bg-primary-600 text-white border-primary-600' 
              : 'bg-white text-calm-700 border-calm-300 hover:bg-calm-50'
          }`}
        >
          {showAll ? 'Show Filtered' : 'Show All'}
        </button>
      </div>

      {/* Workout Grid */}
      {filteredWorkouts.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-calm-200 p-8 text-center">
          <p className="text-calm-700 mb-2">No workouts match your filters.</p>
          <p className="text-sm text-calm-500 mb-4">
            Showing {filteredWorkouts.length} of {workouts.length} workouts
          </p>
          <button
            onClick={() => {
              setSearchTerm('')
              setSelectedTag('')
            }}
            className="text-primary-600 hover:text-primary-700 text-sm underline"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* AI Create Workout Card - Always first */}
          <AICreateWorkoutCard />
          
          {/* Regular workout cards */}
          {filteredWorkouts.map(workout => (
            <WorkoutCard
              key={workout.id}
              workout={workout}
              onComplete={onWorkoutComplete}
              onView={setSelectedWorkout}
            />
          ))}
        </div>
      )}

      <WorkoutDetailModal
        isOpen={!!selectedWorkout}
        onClose={() => setSelectedWorkout(null)}
        workout={selectedWorkout}
        onComplete={onWorkoutComplete}
      />
    </div>
  )
}

export default WorkoutList

