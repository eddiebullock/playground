import { useState, useEffect } from 'react'

export default function Hero() {
  const [videoLoaded, setVideoLoaded] = useState(false)

  useEffect(() => {
    // Preload the video
    const videoUrl = "https://player.vimeo.com/video/1115769247?background=1&autoplay=1&loop=1&byline=0&title=0&muted=1&controls=0&portrait=0&dnt=1&transparent=0"
    
    // Create a temporary iframe to preload
    const tempIframe = document.createElement('iframe')
    tempIframe.src = videoUrl
    tempIframe.style.display = 'none'
    document.body.appendChild(tempIframe)
    
    // Set video as loaded after a short delay
    setTimeout(() => {
      setVideoLoaded(true)
      document.body.removeChild(tempIframe)
    }, 1000)
  }, [])

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Video Background */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-dark-900/30 via-dark-900/50 to-dark-900/70 z-10" />
        <iframe
          src="https://player.vimeo.com/video/1115769247?background=1&autoplay=1&loop=1&byline=0&title=0&muted=1&controls=0&portrait=0&dnt=1&transparent=0"
          className="absolute inset-0 w-full h-full z-20"
          frameBorder="0"
          allow="autoplay; fullscreen; picture-in-picture"
          allowFullScreen
          loading="eager"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: '100vw',
            height: '56.25vw',
            minHeight: '100vh',
            minWidth: '177.78vh',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none'
          }}
        />
        {/* Fallback image if video doesn't load */}
        <img
          src="/images/hero-fallback.jpg"
          alt="Cinematic background"
          className="w-full h-full object-cover absolute inset-0 z-10"
        />
      </div>

      {/* Content Overlay */}
      <div className="relative z-20 text-center px-4 max-w-4xl mx-auto">
        <h1 className="text-5xl md:text-7xl font-serif font-bold text-white mb-6 animate-fade-in">
          Science, Made Accessible.
        </h1>
        <p className="text-xl md:text-2xl text-gray-200 mb-12 max-w-3xl mx-auto leading-relaxed animate-slide-up">
          We turn cutting-edge research into stories the world can see.
        </p>
        
        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-slide-up">
          <button
            onClick={() => scrollToSection('projects')}
            className="bg-primary-500 hover:bg-primary-600 text-white px-8 py-4 rounded-md text-lg font-semibold transition-all duration-300 transform hover:scale-105"
          >
            Watch Our Work
          </button>
          <button
            onClick={() => scrollToSection('contact')}
            className="border-2 border-white text-white hover:bg-white hover:text-dark-900 px-8 py-4 rounded-md text-lg font-semibold transition-all duration-300 transform hover:scale-105"
          >
            Work With Us
          </button>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-20 animate-bounce">
        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </section>
  )
}
