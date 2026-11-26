import { formatDuration, getSensoryLevelColor } from '../utils/helpers'

function WorkoutCard({ workout, onComplete, onView }) {
  const handleComplete = (e) => {
    e.stopPropagation()
    if (onComplete) {
      onComplete(workout.id)
    }
  }

  const handleClick = () => {
    if (onView) {
      onView(workout)
    }
  }

  return (
    <div 
      className="workout-card bg-white rounded-lg shadow-sm border border-calm-200 p-6 hover:shadow-md cursor-pointer transition-all"
      onClick={handleClick}
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-bold text-calm-900">{workout.title}</h3>
        <span className={`px-2 py-1 rounded text-xs font-medium ${getSensoryLevelColor(workout.sensory_level)}`}>
          {workout.sensory_level || 'N/A'}
        </span>
      </div>

      <p className="text-calm-700 mb-4 line-clamp-2">{workout.description}</p>

      <div className="flex flex-wrap gap-2 mb-4">
        {workout.tags && workout.tags.slice(0, 3).map((tag, idx) => (
          <span
            key={idx}
            className="px-2 py-1 bg-calm-100 text-calm-700 text-xs rounded"
          >
            {tag}
          </span>
        ))}
        {workout.tags && workout.tags.length > 3 && (
          <span className="px-2 py-1 bg-calm-100 text-calm-700 text-xs rounded">
            +{workout.tags.length - 3} more
          </span>
        )}
      </div>

      <div className="flex items-center justify-between text-sm text-calm-600 mb-4">
        <span>Duration: {formatDuration(workout.duration || 0)}</span>
        <span>Equipment: {workout.equipment || 'None'}</span>
      </div>

      {onComplete && (
        <button
          onClick={handleComplete}
          className="w-full bg-primary-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors"
          aria-label={`Mark ${workout.title} as complete`}
        >
          Mark Complete
        </button>
      )}
    </div>
  )
}

export default WorkoutCard

