import { useState, useEffect } from 'react'
import { auth, userPreferences } from '../services/api'

function PreferencesModal({ isOpen, onClose, userPrefs, onUpdate }) {
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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (isOpen && userPrefs) {
      setFormData({
        name: userPrefs.name || '',
        sensory_level: userPrefs.sensory_level || '',
        energy_level: userPrefs.energy_level || '',
        environment: userPrefs.environment || '',
        fitness_goal: userPrefs.fitness_goal || '',
        workout_time: userPrefs.workout_time || '',
        equipment: userPrefs.equipment || '',
        special_considerations: userPrefs.special_considerations || '',
      })
    }
  }, [isOpen, userPrefs])

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
        return
      }

      const { data, error: prefError } = await userPreferences.save(user.id, formData)
      
      if (prefError) {
        setError(`Failed to save preferences: ${prefError.message || 'Please try again.'}`)
        console.error('Preferences save error:', prefError)
      } else {
        // Refresh preferences from database to get updated data
        const { data: updatedPrefs } = await userPreferences.get(user.id)
        onUpdate(updatedPrefs || formData)
        onClose()
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
      className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
        selected === value
          ? 'border-primary-500 bg-primary-50'
          : 'border-calm-200 bg-white hover:border-calm-300 hover:bg-calm-50'
      }`}
    >
      <div className="flex items-start">
        <div className={`flex-shrink-0 w-4 h-4 rounded-full border-2 mt-0.5 mr-2 flex items-center justify-center ${
          selected === value ? 'border-primary-600 bg-primary-600' : 'border-calm-300'
        }`}>
          {selected === value && (
            <div className="w-1.5 h-1.5 rounded-full bg-white"></div>
          )}
        </div>
        <div className="flex-1">
          <div className={`text-sm font-medium ${selected === value ? 'text-primary-900' : 'text-calm-900'}`}>
            {label}
          </div>
          {description && (
            <div className={`text-xs mt-0.5 ${selected === value ? 'text-primary-700' : 'text-calm-600'}`}>
              {description}
            </div>
          )}
        </div>
      </div>
    </button>
  )

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div 
          className="fixed inset-0 bg-calm-900 bg-opacity-50 transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        ></div>

        {/* Modal panel */}
        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-calm-900" id="modal-title">
                Preferences
              </h3>
              <button
                onClick={onClose}
                className="text-calm-400 hover:text-calm-600 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md"
                aria-label="Close modal"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {error && (
              <div 
                className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4"
                role="alert"
              >
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
              {/* Name */}
              <div>
                <label 
                  htmlFor="modal-name"
                  className="block text-sm font-semibold text-calm-900 mb-2"
                >
                  Name
                </label>
                <input
                  type="text"
                  id="modal-name"
                  value={formData.name}
                  onChange={(e) => handleChange('name', e.target.value)}
                  placeholder="Enter your name"
                  required
                  className="w-full px-4 py-2 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white"
                />
              </div>

              {/* Fitness Goal */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Fitness Goal
                </label>
                <div className="grid sm:grid-cols-2 gap-2">
                  {[
                    { value: 'strength', label: 'Build Strength', desc: 'Increase muscle strength' },
                    { value: 'flexibility', label: 'Improve Flexibility', desc: 'Increase range of motion' },
                    { value: 'endurance', label: 'Build Endurance', desc: 'Improve cardiovascular fitness' },
                    { value: 'wellness', label: 'General Wellness', desc: 'Maintain overall health' },
                    { value: 'stress', label: 'Stress Relief', desc: 'Reduce anxiety and tension' },
                    { value: 'routine', label: 'Build Routine', desc: 'Establish consistent habits' },
                  ].map((option) => (
                    <OptionCard
                      key={option.value}
                      name="fitness_goal"
                      value={option.value}
                      label={option.label}
                      description={option.desc}
                      selected={formData.fitness_goal}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Sensory Level */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Sensory Sensitivity Level
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'low', label: 'Low', desc: 'Comfortable with various stimuli' },
                    { value: 'medium', label: 'Medium', desc: 'Some sensitivity to noise or brightness' },
                    { value: 'high', label: 'High', desc: 'Need quiet, calm environments' },
                  ].map((option) => (
                    <OptionCard
                      key={option.value}
                      name="sensory_level"
                      value={option.value}
                      label={option.label}
                      description={option.desc}
                      selected={formData.sensory_level}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Energy Level */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Typical Energy Level
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'high', label: 'High', desc: 'Can handle intense workouts' },
                    { value: 'medium', label: 'Medium', desc: 'Moderate intensity preferred' },
                    { value: 'low', label: 'Low', desc: 'Gentle, low-intensity preferred' },
                  ].map((option) => (
                    <OptionCard
                      key={option.value}
                      name="energy_level"
                      value={option.value}
                      label={option.label}
                      description={option.desc}
                      selected={formData.energy_level}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Environment */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Preferred Workout Environment
                </label>
                <div className="grid sm:grid-cols-2 gap-2">
                  {[
                    { value: 'home', label: 'Home', desc: 'Private, controlled space' },
                    { value: 'outdoor', label: 'Outdoor', desc: 'Nature, fresh air' },
                    { value: 'gym', label: 'Gym', desc: 'Structured facility' },
                    { value: 'flexible', label: 'Flexible', desc: 'Anywhere works' },
                  ].map((option) => (
                    <OptionCard
                      key={option.value}
                      name="environment"
                      value={option.value}
                      label={option.label}
                      description={option.desc}
                      selected={formData.environment}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Workout Time */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Preferred Workout Duration
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {['10', '15', '20', '30', '45', '60'].map((time) => (
                    <OptionCard
                      key={time}
                      name="workout_time"
                      value={time}
                      label={time === '60' ? '60+ min' : `${time} min`}
                      selected={formData.workout_time}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Equipment */}
              <div>
                <label className="block text-sm font-semibold text-calm-900 mb-2">
                  Available Equipment
                </label>
                <div className="grid sm:grid-cols-2 gap-2">
                  {[
                    { value: 'none', label: 'None', desc: 'Bodyweight only' },
                    { value: 'basic', label: 'Basic', desc: 'Resistance bands, dumbbells' },
                    { value: 'moderate', label: 'Moderate', desc: 'Some gym equipment' },
                    { value: 'full', label: 'Full', desc: 'Access to full gym' },
                  ].map((option) => (
                    <OptionCard
                      key={option.value}
                      name="equipment"
                      value={option.value}
                      label={option.label}
                      description={option.desc}
                      selected={formData.equipment}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>

              {/* Special Considerations */}
              <div>
                <label 
                  htmlFor="modal-special"
                  className="block text-sm font-semibold text-calm-900 mb-2"
                >
                  Special Considerations (Optional)
                </label>
                <textarea
                  id="modal-special"
                  value={formData.special_considerations}
                  onChange={(e) => handleChange('special_considerations', e.target.value)}
                  rows={3}
                  placeholder="Any specific needs, limitations, or preferences we should know about?"
                  className="w-full px-4 py-2 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none bg-white"
                />
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-primary-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Saving...' : 'Save Changes'}
                </button>
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg border border-calm-300 text-calm-700 hover:bg-calm-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PreferencesModal

