import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { auth, userPreferences } from '../services/api'
import Header from '../components/Header'

function Onboarding() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [formData, setFormData] = useState({
    name: '',
    sensory_level: '',
    energy_level: '',
    environment: '',
    fitness_goal: '',
    workout_time: '',
    equipment: '',
    special_considerations: '',
  })

  useEffect(() => {
    // Check if user is logged in
    const checkUser = async () => {
      const user = await auth.getCurrentUser()
      if (!user) {
        navigate('/auth')
      }
    }
    checkUser()
  }, [navigate])

  const handleChange = (name, value) => {
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const user = await auth.getCurrentUser()
      if (!user) {
        setError('Please log in first')
        navigate('/auth')
        return
      }

      const { error: prefError } = await userPreferences.save(user.id, formData)
      
      if (prefError) {
        setError('Failed to save preferences. Please try again.')
        console.error(prefError)
      } else {
        navigate('/dashboard')
      }
    } catch (err) {
      setError('Something went wrong. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const OptionCard = ({ name, value, label, description, selected, onChange }) => (
    <button
      type="button"
      onClick={() => onChange(name, value)}
      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
        selected === value
          ? 'border-primary-500 bg-primary-50'
          : 'border-calm-200 bg-white hover:border-calm-300 hover:bg-calm-50'
      }`}
    >
      <div className="flex items-start">
        <div className={`flex-shrink-0 w-5 h-5 rounded-full border-2 mt-0.5 mr-3 flex items-center justify-center ${
          selected === value ? 'border-primary-600 bg-primary-600' : 'border-calm-300'
        }`}>
          {selected === value && (
            <div className="w-2 h-2 rounded-full bg-white"></div>
          )}
        </div>
        <div className="flex-1">
          <div className={`font-medium ${selected === value ? 'text-primary-900' : 'text-calm-900'}`}>
            {label}
          </div>
          {description && (
            <div className={`text-sm mt-1 ${selected === value ? 'text-primary-700' : 'text-calm-600'}`}>
              {description}
            </div>
          )}
        </div>
      </div>
    </button>
  )

  return (
    <div className="min-h-screen bg-calm-50">
      <Header />
      
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-white rounded-lg shadow-sm border border-calm-200 p-8">
          <h1 className="text-3xl sm:text-4xl font-bold text-calm-900 mb-2">
            Welcome to NeuroFit
          </h1>
          <p className="text-lg text-calm-700 mb-8">
            Let's customize your fitness experience. This will help us recommend workouts that work for you.
          </p>

          {error && (
            <div 
              className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
              role="alert"
            >
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Name */}
            <div>
              <label 
                htmlFor="name"
                className="block text-base font-semibold text-calm-900 mb-3"
              >
                Name
              </label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                placeholder="Enter your name"
                required
                className="w-full px-4 py-3 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white"
              />
            </div>

            {/* Fitness Goal */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Fitness Goal
              </label>
              <div className="grid sm:grid-cols-2 gap-3">
                <OptionCard
                  name="fitness_goal"
                  value="strength"
                  label="Build Strength"
                  description="Increase muscle strength"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
                <OptionCard
                  name="fitness_goal"
                  value="flexibility"
                  label="Improve Flexibility"
                  description="Increase range of motion"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
                <OptionCard
                  name="fitness_goal"
                  value="endurance"
                  label="Build Endurance"
                  description="Improve cardiovascular fitness"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
                <OptionCard
                  name="fitness_goal"
                  value="wellness"
                  label="General Wellness"
                  description="Maintain overall health"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
                <OptionCard
                  name="fitness_goal"
                  value="stress"
                  label="Stress Relief"
                  description="Reduce anxiety and tension"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
                <OptionCard
                  name="fitness_goal"
                  value="routine"
                  label="Build Routine"
                  description="Establish consistent habits"
                  selected={formData.fitness_goal}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Sensory Level */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Sensory Sensitivity Level
              </label>
              <div className="space-y-3">
                <OptionCard
                  name="sensory_level"
                  value="low"
                  label="Low"
                  description="Comfortable with various stimuli"
                  selected={formData.sensory_level}
                  onChange={handleChange}
                />
                <OptionCard
                  name="sensory_level"
                  value="medium"
                  label="Medium"
                  description="Some sensitivity to noise or brightness"
                  selected={formData.sensory_level}
                  onChange={handleChange}
                />
                <OptionCard
                  name="sensory_level"
                  value="high"
                  label="High"
                  description="Need quiet, calm environments"
                  selected={formData.sensory_level}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Energy Level */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Typical Energy Level
              </label>
              <div className="space-y-3">
                <OptionCard
                  name="energy_level"
                  value="high"
                  label="High"
                  description="Can handle intense workouts"
                  selected={formData.energy_level}
                  onChange={handleChange}
                />
                <OptionCard
                  name="energy_level"
                  value="medium"
                  label="Medium"
                  description="Moderate intensity preferred"
                  selected={formData.energy_level}
                  onChange={handleChange}
                />
                <OptionCard
                  name="energy_level"
                  value="low"
                  label="Low"
                  description="Gentle, low-intensity preferred"
                  selected={formData.energy_level}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Environment */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Preferred Workout Environment
              </label>
              <div className="grid sm:grid-cols-2 gap-3">
                <OptionCard
                  name="environment"
                  value="home"
                  label="Home"
                  description="Private, controlled space"
                  selected={formData.environment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="environment"
                  value="outdoor"
                  label="Outdoor"
                  description="Nature, fresh air"
                  selected={formData.environment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="environment"
                  value="gym"
                  label="Gym"
                  description="Structured facility"
                  selected={formData.environment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="environment"
                  value="flexible"
                  label="Flexible"
                  description="Anywhere works"
                  selected={formData.environment}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Workout Time */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Preferred Workout Duration
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {['10', '15', '20', '30', '45', '60'].map((time) => (
                  <OptionCard
                    key={time}
                    name="workout_time"
                    value={time}
                    label={time === '60' ? '60+ minutes' : `${time} minutes`}
                    selected={formData.workout_time}
                    onChange={handleChange}
                  />
                ))}
              </div>
            </div>

            {/* Equipment */}
            <div>
              <label className="block text-base font-semibold text-calm-900 mb-4">
                Available Equipment
              </label>
              <div className="grid sm:grid-cols-2 gap-3">
                <OptionCard
                  name="equipment"
                  value="none"
                  label="None"
                  description="Bodyweight only"
                  selected={formData.equipment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="equipment"
                  value="basic"
                  label="Basic"
                  description="Resistance bands, dumbbells"
                  selected={formData.equipment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="equipment"
                  value="moderate"
                  label="Moderate"
                  description="Some gym equipment"
                  selected={formData.equipment}
                  onChange={handleChange}
                />
                <OptionCard
                  name="equipment"
                  value="full"
                  label="Full"
                  description="Access to full gym"
                  selected={formData.equipment}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Special Considerations */}
            <div>
              <label 
                htmlFor="special_considerations"
                className="block text-base font-semibold text-calm-900 mb-3"
              >
                Special Considerations (Optional)
              </label>
              <textarea
                id="special_considerations"
                name="special_considerations"
                value={formData.special_considerations}
                onChange={(e) => handleChange('special_considerations', e.target.value)}
                rows={4}
                placeholder="Any specific needs, limitations, or preferences we should know about?"
                className="w-full px-4 py-3 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none bg-white"
              />
            </div>

            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <button
                type="submit"
                disabled={loading || !formData.name || !formData.sensory_level || !formData.energy_level || !formData.environment || !formData.fitness_goal || !formData.workout_time || !formData.equipment}
                className="flex-1 bg-primary-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Saving...' : 'Save Preferences'}
              </button>
              <button
                type="button"
                onClick={() => navigate('/dashboard')}
                className="px-6 py-3 rounded-lg border border-calm-300 text-calm-700 hover:bg-calm-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
              >
                Skip for Now
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Onboarding
