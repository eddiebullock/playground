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
    description: "Full-service documentary production from concept to distribution, specializing in investigative and social impact storytelling.",
    icon: "🎥",
    features: [
      "Investigative research & development",
      "Cinematography & sound design",
      "Post-production & editing",
      "Distribution strategy"
    ]
  },
  {
    id: 2,
    title: "Branded Content & Campaigns",
    description: "Authentic storytelling for brands that want to make a difference, not just sell products.",
    icon: "📺",
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
    description: "Strategic storytelling that helps organizations communicate complex issues and inspire action.",
    icon: "🎙️",
    features: [
      "Story strategy & messaging",
      "Content planning & development",
      "Audience engagement",
      "Campaign optimization"
    ]
  },
  {
    id: 4,
    title: "Impact Media",
    description: "Documentary and media production for NGOs, foundations, and social impact organizations.",
    icon: "🌍",
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
    <section id="services" className="py-20 bg-white">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-bold text-dark-900 mb-6">
            What We Do
          </h2>
          <p className="text-xl text-dark-600 max-w-3xl mx-auto">
            From investigative documentaries to branded content, we create media that matters and drives real change.
          </p>
        </div>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {services.map((service) => (
            <div
              key={service.id}
              className="bg-gray-50 rounded-lg p-8 hover:bg-gray-100 transition-all duration-300 group"
            >
              {/* Service Icon */}
              <div className="text-4xl mb-6 group-hover:scale-110 transition-transform duration-300">
                {service.icon}
              </div>

              {/* Service Title */}
                              <h3 className="text-2xl font-serif font-semibold text-dark-900 mb-4">
                  {service.title}
                </h3>

                {/* Service Description */}
                <p className="text-dark-600 leading-relaxed mb-6">
                  {service.description}
                </p>

                {/* Service Features */}
                <ul className="space-y-2">
                  {service.features.map((feature, index) => (
                    <li key={index} className="flex items-center text-dark-600 text-sm">
                      <svg className="w-4 h-4 text-primary-600 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>

                {/* Learn More Button */}
                <button className="mt-6 text-primary-600 hover:text-primary-700 font-semibold transition-colors duration-200">
                  Learn More →
                </button>
            </div>
          ))}
        </div>

        {/* Process Section */}
        <div className="mt-20 text-center">
          <h3 className="text-3xl font-serif font-bold text-dark-900 mb-12">
            Our Process
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-xl">1</span>
              </div>
              <h4 className="text-lg font-semibold text-dark-900 mb-2">Research</h4>
              <p className="text-dark-600 text-sm">
                Deep investigation and story development
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-xl">2</span>
              </div>
              <h4 className="text-lg font-semibold text-dark-900 mb-2">Production</h4>
              <p className="text-dark-600 text-sm">
                Cinematic filming with authentic storytelling
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-xl">3</span>
              </div>
              <h4 className="text-lg font-semibold text-dark-900 mb-2">Post-Production</h4>
              <p className="text-dark-600 text-sm">
                Editing, sound design, and visual effects
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-xl">4</span>
              </div>
              <h4 className="text-lg font-semibold text-dark-900 mb-2">Distribution</h4>
              <p className="text-dark-600 text-sm">
                Strategic release and impact amplification
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
