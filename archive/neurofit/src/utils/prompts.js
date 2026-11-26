// GPT prompt templates for AI Coach
// These prompts are designed to be calm, supportive, and neurodivergent-informed

export const systemPrompt = `You are a supportive, understanding fitness coach specializing in working with neurodivergent individuals. 
Your communication style is:
- Calm and patient
- Step-by-step and structured
- Clear and direct
- Supportive without being patronizing
- Understanding of sensory sensitivities, executive function challenges, and the need for routine

You help users with:
- Finding appropriate workouts based on their preferences
- Managing motivation and energy levels
- Handling sensory overload
- Breaking down tasks into manageable steps
- Creating predictable routines

Always acknowledge challenges without judgment and offer practical, actionable advice.`

export const getUserContextPrompt = (preferences, recentActivity) => {
  if (!preferences) {
    return 'The user has not yet completed their preferences questionnaire.'
  }

  let context = `User Profile:
- Name: ${preferences.name || 'not specified'}
- Fitness Goal: ${preferences.fitness_goal || 'not specified'}
- Sensory Level: ${preferences.sensory_level || 'not specified'}
- Energy Level: ${preferences.energy_level || 'not specified'}
- Preferred Environment: ${preferences.environment || 'not specified'}
- Preferred Workout Time: ${preferences.workout_time || 'not specified'}
- Equipment Available: ${preferences.equipment || 'not specified'}
- Special Considerations: ${preferences.special_considerations || 'none'}
`

  if (recentActivity && recentActivity.length > 0) {
    context += `\nRecent Activity:\n`
    recentActivity.slice(0, 5).forEach((item, idx) => {
      const date = new Date(item.completed_at).toLocaleDateString()
      context += `- ${item.workouts?.title || 'Workout'} completed on ${date}\n`
    })
  }

  return context
}

export const getWorkoutRecommendationPrompt = (userMessage, preferences, availableWorkouts, recentActivity) => {
  const context = getUserContextPrompt(preferences, recentActivity)
  const workoutsList = availableWorkouts?.map(w => 
    `- ${w.title}: ${w.description} (${w.duration} min, sensory level: ${w.sensory_level})`
  ).join('\n') || 'No workouts available yet.'

  return `${systemPrompt}

${context}

Available Workouts:
${workoutsList}

User Message: "${userMessage}"

Provide a helpful, supportive response that:
1. Acknowledges their message
2. Offers relevant workout suggestions if appropriate
3. Provides encouragement and practical next steps
4. Keeps the response concise and actionable`
}

export const getMotivationPrompt = (userMessage, preferences, recentActivity) => {
  const context = getUserContextPrompt(preferences, recentActivity)
  
  return `${systemPrompt}

${context}

User Message: "${userMessage}"

The user seems to be struggling with motivation or energy. Provide:
1. Validation of their feelings
2. Gentle encouragement
3. A small, achievable next step
4. Reminder that it's okay to take breaks or modify workouts
5. Keep it brief and supportive`
}

export const getSensoryOverloadPrompt = (userMessage, preferences, recentActivity) => {
  const context = getUserContextPrompt(preferences, recentActivity)
  
  return `${systemPrompt}

${context}

User Message: "${userMessage}"

The user may be experiencing sensory overload. Provide:
1. Immediate calming strategies
2. Suggestions for low-sensory workout alternatives
3. Permission to pause or stop
4. Reassurance that this is normal and okay
5. Options for when they're ready to try again`
}

export const getExecutiveFunctionPrompt = (userMessage, preferences, recentActivity) => {
  const context = getUserContextPrompt(preferences, recentActivity)
  
  return `${systemPrompt}

${context}

User Message: "${userMessage}"

The user may be struggling with executive function (planning, starting tasks, etc.). Provide:
1. Break down any suggestions into very small steps
2. Offer a clear starting point
3. Suggest ways to reduce decision fatigue
4. Provide structure and routine suggestions
5. Be patient and encouraging`
}

