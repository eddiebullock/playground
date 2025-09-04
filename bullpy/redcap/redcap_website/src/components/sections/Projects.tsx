import { useState, useEffect } from 'react'

interface Project {
  id: number
  title: string
  description: string
  image: string
  category: string
  year: string
}

const projects: Project[] = [
  {
    id: 1,
    title: "The Race to AGI",
    description: "A feature documentary exploring the global race toward artificial intelligence.",
    image: "https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/5.png",
    category: "Documentary",
    year: "2024"
  },
  {
    id: 2,
    title: "Frontiers",
    description: "Conversations with the public and leading scientists shaping our future.",
    image: "https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/interview.png",
    category: "Series",
    year: "2023"
  },
  {
    id: 3,
    title: "Science Live",
    description: "Talks and workshops bringing science into classrooms and communities.",
    image: "https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/school.png",
    category: "Documentary",
    year: "2024"
  },
  {
    id: 4,
    title: "Articles & Newsletter",
    description: "Sharp, accessible writing that translates research into everyday insight.",
    image: "https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/school.png",
    category: "Series",
    year: "2023"
  }
]

export default function Projects() {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [touchStart, setTouchStart] = useState<number | null>(null)
  const [touchEnd, setTouchEnd] = useState<number | null>(null)

  const nextProject = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % projects.length)
  }

  const prevProject = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + projects.length) % projects.length)
  }

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null)
    setTouchStart(e.targetTouches[0].clientX)
  }

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX)
  }

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return
    
    const distance = touchStart - touchEnd
    const isLeftSwipe = distance > 50
    const isRightSwipe = distance < -50

    if (isLeftSwipe) {
      nextProject()
    }
    if (isRightSwipe) {
      prevProject()
    }
  }

  // Add keyboard navigation for accessibility
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        prevProject()
      } else if (e.key === 'ArrowRight') {
        nextProject()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <section id="projects" className="py-20 bg-white">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-bold text-dark-900 mb-6">
            Our Projects
          </h2>
          <p className="text-xl text-dark-600 max-w-3xl mx-auto">
            Stories that matter, told with purpose and passion. Each project represents our commitment to truth and impact.
          </p>
        </div>

        {/* Desktop Grid */}
        <div className="hidden md:grid md:grid-cols-2 gap-8">
          {projects.map((project) => (
            <div
              key={project.id}
              className="group relative overflow-hidden rounded-lg bg-gray-50 hover:bg-gray-100 transition-all duration-300 cursor-pointer"
              onClick={() => setSelectedProject(project)}
            >
              {/* Project Image */}
              <div className="relative h-80 overflow-hidden">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-dark-900/80 via-transparent to-transparent" />
                
                {/* Play Button Overlay */}
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className="bg-primary-500 hover:bg-primary-600 w-16 h-16 rounded-full flex items-center justify-center transition-colors duration-300">
                    <svg className="w-6 h-6 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </div>
                </div>
              </div>

              {/* Project Info */}
              <div className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-primary-600 font-semibold">
                    {project.category}
                  </span>
                  <span className="text-sm text-dark-400">
                    {project.year}
                  </span>
                </div>
                <h3 className="text-xl font-serif font-semibold text-dark-900 mb-3">
                  {project.title}
                </h3>
                <p className="text-dark-600 text-sm leading-relaxed">
                  {project.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Mobile Carousel */}
        <div className="md:hidden relative">
          <div className="overflow-hidden rounded-lg">
            <div 
              className="flex transition-transform duration-300 ease-in-out"
              style={{ transform: `translateX(-${currentIndex * 100}%)` }}
              onTouchStart={onTouchStart}
              onTouchMove={onTouchMove}
              onTouchEnd={onTouchEnd}
            >
              {projects.map((project) => (
                <div key={project.id} className="w-full flex-shrink-0">
                  <div
                    className="group relative overflow-hidden rounded-lg bg-gray-50 cursor-pointer mx-4"
                    onClick={() => setSelectedProject(project)}
                  >
                    {/* Project Image */}
                    <div className="relative h-80 overflow-hidden">
                      <img
                        src={project.image}
                        alt={project.title}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-dark-900/80 via-transparent to-transparent" />
                      
                      {/* Play Button Overlay */}
                      <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <div className="bg-primary-500 hover:bg-primary-600 w-16 h-16 rounded-full flex items-center justify-center transition-colors duration-300">
                          <svg className="w-6 h-6 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M8 5v14l11-7z" />
                          </svg>
                        </div>
                      </div>
                    </div>

                    {/* Project Info */}
                    <div className="p-6">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-primary-600 font-semibold">
                          {project.category}
                        </span>
                        <span className="text-sm text-dark-400">
                          {project.year}
                        </span>
                      </div>
                      <h3 className="text-xl font-serif font-semibold text-dark-900 mb-3">
                        {project.title}
                      </h3>
                      <p className="text-dark-600 text-sm leading-relaxed">
                        {project.description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Navigation Dots */}
          <div className="flex justify-center mt-6 space-x-2">
            {projects.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={`w-3 h-3 rounded-full transition-colors duration-200 ${
                  index === currentIndex ? 'bg-primary-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>

          {/* Swipe Instructions */}
          <div className="text-center mt-4">
            <div className="flex items-center justify-center space-x-2 text-sm text-dark-400">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span>Swipe to navigate</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </div>
        </div>

        {/* View All Projects Button */}
        <div className="text-center mt-12">
          <button className="border-2 border-primary-500 text-primary-600 hover:bg-primary-500 hover:text-white px-8 py-3 rounded-md font-semibold transition-all duration-300">
            View All Projects
          </button>
        </div>
      </div>

      {/* Project Modal */}
      {selectedProject && (
        <div className="fixed inset-0 bg-dark-900/95 z-50 flex items-center justify-center p-4">
          <div className="bg-dark-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="relative">
              <button
                onClick={() => setSelectedProject(null)}
                className="absolute top-4 right-4 text-white hover:text-primary-400 z-10"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              
              <img
                src={selectedProject.image}
                alt={selectedProject.title}
                className="w-full h-64 object-cover rounded-t-lg"
              />
              
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-primary-400 font-semibold">
                    {selectedProject.category}
                  </span>
                  <span className="text-gray-400">
                    {selectedProject.year}
                  </span>
                </div>
                <h3 className="text-2xl font-serif font-bold text-white mb-4">
                  {selectedProject.title}
                </h3>
                <p className="text-gray-300 leading-relaxed mb-6">
                  {selectedProject.description}
                </p>
                <p className="text-gray-400 text-sm">
                  [Placeholder: Full project description, synopsis, and trailer would go here. This is where you'd add the complete story, key themes, and impact of the project.]
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
