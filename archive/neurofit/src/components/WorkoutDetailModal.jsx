import { formatDuration, getSensoryLevelColor } from '../utils/helpers'

function WorkoutDetailModal({ isOpen, onClose, workout, onComplete }) {
  if (!isOpen || !workout) return null

  const handleComplete = () => {
    if (onComplete) {
      onComplete(workout.id)
      onClose()
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="workout-title" role="dialog" aria-modal="true">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div 
          className="fixed inset-0 bg-calm-900 bg-opacity-50 transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        ></div>

        {/* Modal panel */}
        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full">
          <div className="bg-white px-6 pt-6 pb-6 sm:p-8">
            {/* Header */}
            <div className="flex items-start justify-between mb-6">
              <div className="flex-1 pr-4">
                <h3 className="text-3xl font-bold text-calm-900 mb-3" id="workout-title">
                  {workout.title}
                </h3>
                <div className="flex flex-wrap items-center gap-3 mb-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSensoryLevelColor(workout.sensory_level)}`}>
                    {workout.sensory_level ? `${workout.sensory_level.charAt(0).toUpperCase() + workout.sensory_level.slice(1)} Sensory` : 'N/A'}
                  </span>
                  <span className="px-3 py-1 bg-calm-100 text-calm-700 rounded-full text-sm font-medium">
                    {formatDuration(workout.duration || 0)}
                  </span>
                  <span className="px-3 py-1 bg-calm-100 text-calm-700 rounded-full text-sm font-medium">
                    {workout.equipment === 'none' ? 'No Equipment' : workout.equipment ? workout.equipment.charAt(0).toUpperCase() + workout.equipment.slice(1) : 'None'}
                  </span>
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-calm-400 hover:text-calm-600 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-md flex-shrink-0"
                aria-label="Close modal"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Description */}
            <div className="mb-6">
              <h4 className="text-base font-semibold text-calm-900 mb-3">About This Workout</h4>
              <p className="text-calm-700 leading-relaxed text-base">{workout.description}</p>
            </div>

            {/* Tags */}
            {workout.tags && workout.tags.length > 0 && (
              <div className="mb-6">
                <h4 className="text-base font-semibold text-calm-900 mb-3">Focus Areas</h4>
                <div className="flex flex-wrap gap-2">
                  {workout.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1.5 bg-primary-100 text-primary-800 text-sm rounded-full font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 pt-6 border-t border-calm-200">
              {onComplete && (
                <button
                  onClick={handleComplete}
                  className="flex-1 bg-primary-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
                >
                  Mark as Complete
                </button>
              )}
              <button
                onClick={onClose}
                className="px-6 py-3 rounded-lg border border-calm-300 text-calm-700 hover:bg-calm-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WorkoutDetailModal

