import { useState, useEffect } from 'react'

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY
      setIsScrolled(scrollPosition > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
    setIsMenuOpen(false)
  }

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      isScrolled 
        ? 'bg-white/95 backdrop-blur-sm border-b border-gray-200 shadow-lg' 
        : 'bg-transparent'
    }`}>
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center">
            <div className="w-8 h-8 bg-primary-500 rounded-full mr-3 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </div>
            <h1 className={`text-2xl font-serif font-bold transition-colors duration-300 ${
              isScrolled ? 'text-dark-900' : 'text-white'
            }`}>
              RedCap Media
            </h1>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => scrollToSection('projects')}
              className={`transition-colors duration-200 ${
                isScrolled 
                  ? 'text-dark-600 hover:text-primary-600' 
                  : 'text-white hover:text-primary-400'
              }`}
            >
              Projects
            </button>
            <button
              onClick={() => scrollToSection('about')}
              className={`transition-colors duration-200 ${
                isScrolled 
                  ? 'text-dark-600 hover:text-primary-600' 
                  : 'text-white hover:text-primary-400'
              }`}
            >
              About
            </button>
            <button
              onClick={() => scrollToSection('services')}
              className={`transition-colors duration-200 ${
                isScrolled 
                  ? 'text-dark-600 hover:text-primary-600' 
                  : 'text-white hover:text-primary-400'
              }`}
            >
              Services
            </button>
            <button
              onClick={() => scrollToSection('contact')}
              className="bg-primary-500 hover:bg-primary-600 text-white px-6 py-2 rounded-md transition-colors duration-200"
            >
              Contact
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className={`md:hidden transition-colors duration-200 ${
              isScrolled 
                ? 'text-dark-600 hover:text-primary-600' 
                : 'text-white hover:text-primary-400'
            }`}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <nav className={`md:hidden mt-4 pb-4 border-t transition-colors duration-200 ${
            isScrolled ? 'border-gray-200' : 'border-white/20'
          }`}>
            <div className="flex flex-col space-y-4 pt-4">
              <button
                onClick={() => scrollToSection('projects')}
                className={`transition-colors duration-200 text-left ${
                  isScrolled 
                    ? 'text-dark-600 hover:text-primary-600' 
                    : 'text-white hover:text-primary-400'
                }`}
              >
                Projects
              </button>
              <button
                onClick={() => scrollToSection('about')}
                className={`transition-colors duration-200 text-left ${
                  isScrolled 
                    ? 'text-dark-600 hover:text-primary-600' 
                    : 'text-white hover:text-primary-400'
                }`}
              >
                About
              </button>
              <button
                onClick={() => scrollToSection('services')}
                className={`transition-colors duration-200 text-left ${
                  isScrolled 
                    ? 'text-dark-600 hover:text-primary-600' 
                    : 'text-white hover:text-primary-400'
                }`}
              >
                Services
              </button>
              <button
                onClick={() => scrollToSection('contact')}
                className="bg-primary-500 hover:bg-primary-600 text-white px-6 py-2 rounded-md transition-colors duration-200 text-left"
              >
                Contact
              </button>
            </div>
          </nav>
        )}
      </div>
    </header>
  )
}
