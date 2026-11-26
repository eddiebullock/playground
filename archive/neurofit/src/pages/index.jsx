import { useState } from 'react'
import Header from '../components/Header'
import Footer from '../components/Footer'

function LandingPage() {
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState('')

  const validateEmail = (email) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    setError('')
    
    if (!email.trim()) {
      setError('Please enter your email address')
      return
    }
    
    if (!validateEmail(email)) {
      setError('Please enter a valid email address')
      return
    }
    
    // In a real app, this would send to your backend
    console.log('Email submitted:', email)
    setSubmitted(true)
    setEmail('')
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      {/* Hero Section */}
      <section 
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-24 lg:py-32"
        aria-labelledby="hero-heading"
      >
        <div className="text-center max-w-3xl mx-auto">
          <h1 
            id="hero-heading"
            className="text-4xl sm:text-5xl lg:text-6xl font-bold text-calm-900 mb-6 leading-tight"
          >
            Fitness designed for neurodivergent minds
          </h1>
          <p className="text-xl sm:text-2xl text-calm-700 mb-12 leading-relaxed">
            Structured workouts, AI coach guidance, and low-stress routines for autistic and neurodivergent users
          </p>
        </div>
      </section>

      {/* Problem Section */}
      <section 
        className="bg-white py-16 sm:py-24"
        aria-labelledby="problem-heading"
      >
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 
            id="problem-heading"
            className="text-3xl sm:text-4xl font-bold text-calm-900 mb-8 text-center"
          >
            Why NeuroFit?
          </h2>
          <div className="prose prose-lg max-w-none text-calm-700 space-y-6">
            <p className="text-lg leading-relaxed">
              Traditional gyms can be overwhelming for neurodivergent individuals. Bright lights, loud music, 
              crowded spaces, and social expectations create barriers that make fitness inaccessible.
            </p>
            <p className="text-lg leading-relaxed">
              Many neurodivergent people struggle with sensory sensitivities, social anxiety, and the need for 
              predictable routines. These challenges make it difficult to access the physical and mental health 
              benefits of regular exercise.
            </p>
            <p className="text-lg leading-relaxed">
              NeuroFit removes these barriers by providing structured, predictable workouts in a low-stress 
              environment, with AI guidance that adapts to your needs and preferences.
            </p>
          </div>
        </div>
      </section>

      {/* Waitlist CTA Section */}
      <section 
        id="waitlist"
        className="bg-calm-100 py-16 sm:py-24"
        aria-labelledby="waitlist-heading"
      >
        <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 
            id="waitlist-heading"
            className="text-3xl sm:text-4xl font-bold text-calm-900 mb-4 text-center"
          >
            Join the Waitlist
          </h2>
          <p className="text-lg text-calm-700 mb-8 text-center">
            Be among the first to experience fitness designed for you
          </p>
          
          {submitted ? (
            <div 
              className="bg-green-50 border border-green-200 rounded-lg p-6 text-center"
              role="alert"
              aria-live="polite"
            >
              <p className="text-green-800 font-medium">
                Thank you! We'll be in touch soon.
              </p>
            </div>
          ) : (
            <form 
              onSubmit={handleSubmit}
              className="space-y-4"
              aria-label="Waitlist signup form"
            >
              <div>
                <label 
                  htmlFor="email"
                  className="block text-sm font-medium text-calm-700 mb-2 sr-only"
                >
                  Email address
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value)
                    setError('')
                  }}
                  placeholder="Enter your email"
                  required
                  aria-required="true"
                  aria-invalid={error ? 'true' : 'false'}
                  aria-describedby={error ? 'email-error' : undefined}
                  className="w-full px-4 py-3 rounded-lg border border-calm-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-calm-900 placeholder-calm-400 transition-all"
                />
                {error && (
                  <p 
                    id="email-error"
                    className="mt-2 text-sm text-red-600"
                    role="alert"
                  >
                    {error}
                  </p>
                )}
              </div>
              <button
                type="submit"
                className="w-full bg-primary-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Submit email to join waitlist"
              >
                Join Waitlist
              </button>
            </form>
          )}
        </div>
      </section>

      {/* Testimonials Section */}
      <section 
        className="py-16 sm:py-24"
        aria-labelledby="testimonials-heading"
      >
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 
            id="testimonials-heading"
            className="text-3xl sm:text-4xl font-bold text-calm-900 mb-12 text-center"
          >
            What People Are Saying
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Placeholder testimonials */}
            <div className="bg-white rounded-lg shadow-sm p-6 border border-calm-200">
              <p className="text-calm-700 mb-4 italic">
                "Testimonials coming soon..."
              </p>
              <p className="text-sm text-calm-500 font-medium">
                — Early Access User
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-sm p-6 border border-calm-200">
              <p className="text-calm-700 mb-4 italic">
                "Testimonials coming soon..."
              </p>
              <p className="text-sm text-calm-500 font-medium">
                — Early Access User
              </p>
            </div>
            <div className="bg-white rounded-lg shadow-sm p-6 border border-calm-200 md:col-span-2 lg:col-span-1">
              <p className="text-calm-700 mb-4 italic">
                "Testimonials coming soon..."
              </p>
              <p className="text-sm text-calm-500 font-medium">
                — Early Access User
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}

export default LandingPage

