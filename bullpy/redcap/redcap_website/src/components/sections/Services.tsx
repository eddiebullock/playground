interface Service {
  id: number
  title: string
  description: string
  icon: string
  features: string[]
}

const services: Service[] = [
  {
    id: 1,
    title: "Documentary Production",
    description: "Full-service documentary production from concept to distribution, specialising in scientific storytelling.",
    icon: "video",
    features: [
      "Investigative research & development",
      "Cinematography & sound design",
      "Post-production & editing",
      "Distribution strategy"
    ]
  },
  {
    id: 2,
    title: "Social Media",
    description: "Social media is a huge opportunity to disseminate science. We create engaging content that makes research accessible.",
    icon: "megaphone",
    features: [
      "Brand narrative development",
      "Campaign video production",
      "Social media content",
      "Impact measurement"
    ]
  },
  {
    id: 3,
    title: "Creative Strategy & Storytelling",
    description: "Strategic storytelling that helps organisations communicate complex issues and inspire action.",
    icon: "lightbulb",
    features: [
      "Story strategy & messaging",
      "Content planning & development",
      "Audience engagement",
      "Campaign optimisation"
    ]
  },
  {
    id: 4,
    title: "Science Communications",
    description: "Media production for science organisations, universities, tech companies and research institutions.",
    icon: "globe",
    features: [
      "Impact storytelling",
      "Advocacy video production",
      "Grant proposal support",
      "Partnership development"
    ]
  }
]

export default function Services() {
  return (
    <>
      <section id="services" className="py-20 bg-white relative overflow-hidden">
        {/* Background Image */}
        <div className="absolute inset-0 z-0">
          <img 
            src="https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/WhatsApp%20Image%202025-08-31%20at%2015.58.40%20(1).jpeg" 
            alt="Background - What we do" 
            className="w-full h-full object-cover opacity-60"
          />
        </div>
        
        <div className="container mx-auto px-4 relative z-10">
          {/* Section Header */}
          <div className="text-center mb-16">
                      <h2 className="text-4xl md:text-5xl font-serif font-bold text-dark-900 mb-6">
            What We Do
          </h2>
          <p className="text-xl text-dark-600 max-w-3xl mx-auto">
            From documentary production to social media content, we create science communications that reach real people.
          </p>
          </div>

          {/* Services Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {services.map((service) => (
              <div
                key={service.id}
                className="text-center p-8 hover:bg-white/50 transition-all duration-300 group rounded-lg"
              >
                              {/* Service Icon */}
              <div className="mb-6 group-hover:scale-110 transition-transform duration-300">
                {service.icon === 'video' && (
                  <svg className="w-16 h-16 text-primary-600 mx-auto" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
                  </svg>
                )}
                {service.icon === 'megaphone' && (
                  <svg className="w-16 h-16 text-primary-600 mx-auto" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                )}
                {service.icon === 'lightbulb' && (
                  <svg className="w-16 h-16 text-primary-600 mx-auto" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7zm-1.5 13v2h-1v-2h1zm-1-3h1v1h-1v-1z"/>
                  </svg>
                )}
                {service.icon === 'globe' && (
                  <svg className="w-16 h-16 text-primary-600 mx-auto" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                  </svg>
                )}
              </div>

                {/* Service Title */}
                <h3 className="text-2xl font-serif font-semibold text-dark-900 mb-4">
                  {service.title}
                </h3>

                {/* Service Description */}
                <p className="text-dark-600 leading-relaxed mb-6">
                  {service.description}
                </p>
              </div>
            ))}
          </div>

          {/* Work With Us Button */}
          <div className="mt-16 text-center">
            <button className="btn-primary text-lg px-8 py-4">
              Work With Us
            </button>
          </div>
        </div>
      </section>

      {/* Process Section - Separate Full Width */}
      <section className="py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <h3 className="text-3xl font-serif font-bold text-dark-900 mb-12">
              Our Process
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                  </svg>
                </div>
                <h4 className="text-lg font-semibold text-dark-900 mb-2">1. Research</h4>
                <p className="text-dark-600 text-sm">
                  We dig into the latest science with academic rigour.
                </p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                  </svg>
                </div>
                <h4 className="text-lg font-semibold text-dark-900 mb-2">2. Storytelling</h4>
                <p className="text-dark-600 text-sm">
                  We translate complexity into clear, compelling narratives.
                </p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                </div>
                <h4 className="text-lg font-semibold text-dark-900 mb-2">3. Impact</h4>
                <p className="text-dark-600 text-sm">
                  We share it through film, talks, and writing to reach real people.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
