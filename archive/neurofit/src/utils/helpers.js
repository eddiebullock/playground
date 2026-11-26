// Helper utility functions

// Format date for display
export const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

// Format duration (minutes to readable format)
export const formatDuration = (minutes) => {
  if (minutes < 60) {
    return `${minutes} min`
  }
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`
}

// Debounce function for search/input
export const debounce = (func, wait) => {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

// Filter workouts by preferences
export const filterWorkoutsByPreferences = (workouts, preferences) => {
  if (!preferences || !workouts || workouts.length === 0) {
    console.log('No filtering - missing preferences or workouts')
    return workouts || []
  }

  console.log('Filtering workouts with preferences:', preferences)
  console.log('Total workouts before filter:', workouts.length)

  const filtered = workouts.filter(workout => {
    // Filter by sensory level if specified
    if (preferences.sensory_level && workout.sensory_level) {
      const userSensory = preferences.sensory_level.toLowerCase()
      const workoutSensory = workout.sensory_level.toLowerCase()
      
      // Allow workouts that match or are lower sensory
      // Low users: only low workouts
      // Medium users: low or medium workouts
      // High users: all workouts
      if (userSensory === 'low' && workoutSensory !== 'low') {
        console.log(`Filtered out ${workout.title}: sensory mismatch (user: ${userSensory}, workout: ${workoutSensory})`)
        return false
      }
      if (userSensory === 'medium' && workoutSensory === 'high') {
        console.log(`Filtered out ${workout.title}: sensory mismatch (user: ${userSensory}, workout: ${workoutSensory})`)
        return false
      }
    }

    // Filter by equipment if specified
    if (preferences.equipment && workout.equipment) {
      const userEquipment = preferences.equipment.toLowerCase()
      const workoutEquipment = workout.equipment.toLowerCase()
      
      // If user has no equipment, only show workouts that need no equipment
      if (userEquipment === 'none' && workoutEquipment !== 'none') {
        console.log(`Filtered out ${workout.title}: equipment mismatch (user: ${userEquipment}, workout: ${workoutEquipment})`)
        return false
      }
    }

    return true
  })

  console.log('Workouts after filtering:', filtered.length)
  return filtered
}

// Get sensory level color
export const getSensoryLevelColor = (level) => {
  const levels = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
  }
  return levels[level?.toLowerCase()] || 'bg-calm-100 text-calm-800'
}

// Validate email
export const validateEmail = (email) => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
}

// Validate password (minimum 6 characters)
export const validatePassword = (password) => {
  return password && password.length >= 6
}

