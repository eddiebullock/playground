// API helper functions for database operations
import { supabase } from './supabaseClient'

// Auth functions
export const auth = {
  signUp: async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    return { data, error }
  },

  signIn: async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { data, error }
  },

  signOut: async () => {
    const { error } = await supabase.auth.signOut()
    return { error }
  },

  getCurrentUser: async () => {
    const { data: { user } } = await supabase.auth.getUser()
    return user
  },

  getSession: async () => {
    const { data: { session } } = await supabase.auth.getSession()
    return session
  },
  
  supabase, // Export supabase for auth state changes
}

// User preferences functions
export const userPreferences = {
  save: async (userId, preferences) => {
    const { data, error } = await supabase
      .from('profiles')
      .upsert({
        user_id: userId,
        ...preferences,
        updated_at: new Date().toISOString(),
      })
    return { data, error }
  },

  get: async (userId) => {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('user_id', userId)
      .single()
    return { data, error }
  },
}

// Workout functions
export const workouts = {
  getAll: async () => {
    const { data, error } = await supabase
      .from('workouts')
      .select('*')
      .order('created_at', { ascending: false })
    return { data, error }
  },

  getById: async (id) => {
    const { data, error } = await supabase
      .from('workouts')
      .select('*')
      .eq('id', id)
      .single()
    return { data, error }
  },

  getByTags: async (tags) => {
    const { data, error } = await supabase
      .from('workouts')
      .select('*')
      .contains('tags', tags)
    return { data, error }
  },

  create: async (workout) => {
    const { data, error } = await supabase
      .from('workouts')
      .insert([workout])
      .select()
      .single()
    return { data, error }
  },
}

// Progress tracking functions
export const progress = {
  logWorkout: async (userId, workoutId) => {
    const { data, error } = await supabase
      .from('progress')
      .insert({
        user_id: userId,
        workout_id: workoutId,
        completed_at: new Date().toISOString(),
      })
    return { data, error }
  },

  getUserProgress: async (userId) => {
    const { data, error } = await supabase
      .from('progress')
      .select(`
        *,
        workouts (*)
      `)
      .eq('user_id', userId)
      .order('completed_at', { ascending: false })
    return { data, error }
  },

  getStats: async (userId) => {
    const { data, error } = await supabase
      .from('progress')
      .select('*', { count: 'exact', head: false })
      .eq('user_id', userId)
      .order('completed_at', { ascending: false })
    
    if (error) return { stats: null, error }
    
    const totalWorkouts = data?.length || 0
    const thisWeek = data?.filter(item => {
      const completedDate = new Date(item.completed_at)
      const weekAgo = new Date()
      weekAgo.setDate(weekAgo.getDate() - 7)
      return completedDate >= weekAgo
    }).length || 0

    // Calculate streak
    let streak = 0
    if (data && data.length > 0) {
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      
      let checkDate = new Date(today)
      let foundToday = false
      
      for (let i = 0; i < data.length; i++) {
        const completedDate = new Date(data[i].completed_at)
        completedDate.setHours(0, 0, 0, 0)
        
        const daysDiff = Math.floor((checkDate - completedDate) / (1000 * 60 * 60 * 24))
        
        if (daysDiff === 0 && !foundToday) {
          foundToday = true
          streak++
          checkDate.setDate(checkDate.getDate() - 1)
        } else if (daysDiff === 1) {
          streak++
          checkDate.setDate(checkDate.getDate() - 1)
        } else if (daysDiff > 1) {
          break
        }
      }
    }

    const lastCompleted = data && data.length > 0 ? data[0].completed_at : null

    return {
      stats: {
        totalWorkouts,
        thisWeek,
        streak,
        lastCompleted,
      },
      error: null,
    }
  },
}

