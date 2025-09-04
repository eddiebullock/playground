interface TeamMember {
  id: number
  name: string
  role: string
  bio: string
  image: string
}

const teamMembers: TeamMember[] = [
  {
    id: 1,
    name: "Sarah Chen",
    role: "Lead Presenter & Director",
    bio: "Award-winning journalist with 15 years of experience covering social justice issues and human rights stories across the globe.",
    image: "/images/team/sarah-chen.jpg"
  },
  {
    id: 2,
    name: "Marcus Rodriguez",
    role: "Executive Producer",
    bio: "Former investigative reporter turned filmmaker, specializing in long-form documentaries that challenge conventional narratives.",
    image: "/images/team/marcus-rodriguez.jpg"
  },
  {
    id: 3,
    name: "Aisha Patel",
    role: "Creative Director",
    bio: "Visual storyteller with a background in photojournalism and documentary filmmaking, focused on amplifying marginalized voices.",
    image: "/images/team/aisha-patel.jpg"
  }
]

export default function About() {
  return (
    <section id="about" className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Mission Statement */}
          <div>
            <h2 className="text-4xl md:text-5xl font-serif font-bold text-dark-900 mb-8">
              About RedCap Media
            </h2>
            <p className="text-xl md:text-2xl text-dark-600 leading-relaxed mb-8">
              We believe in the power of stories to shift cultures. Our team blends investigative journalism, cinematic craft, and creative strategy to bring hidden realities to light.
            </p>
            <p className="text-lg text-dark-500 leading-relaxed mb-8">
              For over a decade, we've been telling stories that matter—stories that challenge perspectives, amplify voices, and inspire real change. From intimate character studies to sweeping investigative pieces, we approach each project with the same commitment to truth, integrity, and impact.
            </p>
          </div>

          {/* BTS Image/Video */}
          <div className="relative">
            <div className="aspect-video bg-gray-200 rounded-lg overflow-hidden">
              <img
                src="/images/about/bts-photo.jpg"
                alt="Behind the scenes of our documentary production"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-dark-900/60 via-transparent to-transparent" />
              <div className="absolute bottom-6 left-6">
                <p className="text-white text-sm font-medium">
                  Behind the Scenes: Filming "Voices of the Valley"
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Team Section */}
        <div className="text-center mb-16 mt-20">
          <h3 className="text-3xl md:text-4xl font-serif font-bold text-dark-900 mb-6">
            Meet Our Team
          </h3>
          <p className="text-xl text-dark-600 max-w-3xl mx-auto">
            The passionate storytellers behind every RedCap Media project.
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {teamMembers.map((member) => (
            <div key={member.id} className="text-center group">
              {/* Team Member Image */}
              <div className="relative mb-6 overflow-hidden rounded-lg">
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-full h-80 object-cover group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-dark-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>

              {/* Team Member Info */}
              <h4 className="text-xl font-serif font-semibold text-dark-900 mb-2">
                {member.name}
              </h4>
              <p className="text-primary-600 font-medium mb-4">
                {member.role}
              </p>
              <p className="text-dark-600 text-sm leading-relaxed">
                {member.bio}
              </p>
            </div>
          ))}
        </div>

        {/* Stats Section */}
        <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div>
            <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
              50+
            </div>
            <div className="text-dark-500 text-sm">
              Documentaries Produced
            </div>
          </div>
          <div>
            <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
              25+
            </div>
            <div className="text-dark-500 text-sm">
              Countries Filmed In
            </div>
          </div>
          <div>
            <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
              15+
            </div>
            <div className="text-dark-500 text-sm">
              Awards Won
            </div>
          </div>
          <div>
            <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
              10M+
            </div>
            <div className="text-dark-500 text-sm">
              Views Worldwide
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
