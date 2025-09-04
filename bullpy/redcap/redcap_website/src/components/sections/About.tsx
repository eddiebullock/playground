import React from 'react';

const About: React.FC = () => {
  return (
    <section id="about" className="section-padding bg-gray-50">
      <div className="container mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif text-dark-900 mb-4">
            Our Story
          </h2>
          <div className="w-24 h-1 bg-yellow-400 mx-auto"></div>
        </div>

        {/* About Content */}
        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* Text Content */}
          <div className="space-y-6">
            <h2 className="text-4xl md:text-5xl font-serif text-dark-900 mb-6">
              About RedCap Media
            </h2>
            <p className="text-lg text-dark-600 leading-relaxed">
              There's a huge gap between the science published in academic journals and the public. Our mission 
              is to solve this problem by disseminating scientific information in an appealing, accessible way.
            </p>
            <p className="text-lg text-dark-600 leading-relaxed">
              Our belief is simple: knowledge shouldn't stay locked in journals — it should reach people 
              where they are, in ways that resonate and inspire action.
            </p>
          </div>

          {/* Behind the Scenes Collage */}
          <div className="relative">
            {/* Video and Image Side by Side */}
            <div className="flex gap-4 mb-4">
              {/* Main Video */}
              <div className="relative flex-1">
                <div style={{padding:'56.25% 0 0 0',position:'relative'}} className="rounded-lg overflow-hidden">
                  <iframe 
                    src="https://player.vimeo.com/video/1115778704?background=1&autoplay=1&loop=1&byline=0&title=0&muted=1&controls=0&portrait=0&dnt=1" 
                    frameBorder="0" 
                    allow="autoplay; fullscreen; picture-in-picture" 
                    loading="lazy"
                    style={{position:'absolute',top:0,left:0,width:'100%',height:'100%',borderRadius:'8px'}} 
                    title="rcm about"
                  />
                </div>
              </div>
              
              {/* Image next to video */}
              <div className="relative group overflow-hidden rounded-lg w-1/3">
                <img 
                  src="/images/about/bts-1.jpg" 
                  alt="Behind the scenes - Production setup" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  style={{height:'100%'}}
                />
                <div className="absolute inset-0 bg-black bg-opacity-20 group-hover:bg-opacity-0 transition-all duration-300" />
              </div>
            </div>
            
            {/* Bottom row - 2 images */}
            <div className="flex gap-3">
              <div className="relative group overflow-hidden rounded-lg flex-1">
                <img 
                  src="/images/about/bts-2.jpg" 
                  alt="Behind the scenes - Camera work" 
                  className="w-full h-24 md:h-32 object-cover transition-transform duration-300 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-black bg-opacity-20 group-hover:bg-opacity-0 transition-all duration-300" />
              </div>
              <div className="relative group overflow-hidden rounded-lg flex-1">
                <img 
                  src="/images/about/bts-3.jpg" 
                  alt="Behind the scenes - Team collaboration" 
                  className="w-full h-24 md:h-32 object-cover transition-transform duration-300 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-black bg-opacity-20 group-hover:bg-opacity-0 transition-all duration-300" />
              </div>
            </div>
          </div>
        </div>

        {/* Team Section */}
        <div className="mt-20">
          <h3 className="text-3xl font-serif text-dark-900 text-center mb-12">
            Meet Our Team
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
                        {/* Team Member 1 - Eddie */}
            <div className="text-center group">
              <div className="relative mb-6">
                <div className="w-48 h-48 mx-auto bg-gray-700 rounded-full overflow-hidden">
                  <img 
                    src="/images/team/ed-head.png" 
                    alt="Eddie" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Eddie</h4>
              <p className="text-primary-600 mb-3">Creative Director & Producer</p>
              <p className="text-dark-600 text-sm">
                Award-winning filmmaker with a passion for uncovering untold stories and amplifying marginalized voices.
              </p>
            </div>

            {/* Team Member 2 - Alf */}
            <div className="text-center group">
              <div className="relative mb-6">
                <div className="w-48 h-48 mx-auto bg-gray-700 rounded-full overflow-hidden">
                  <img 
                    src="/images/team/alf-head.png" 
                    alt="Alf" 
                    className="w-full h-full object-cover scale-110"
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Alf</h4>
              <p className="text-primary-600 mb-3">Director of Photography</p>
              <p className="text-dark-600 text-sm">
                Visual storyteller who transforms complex narratives into compelling cinematic experiences.
              </p>
            </div>

            {/* Team Member 3 - Sammy */}
            <div className="text-center group">
              <div className="relative mb-6">
                <div className="w-48 h-48 mx-auto bg-gray-700 rounded-full overflow-hidden">
                  <img 
                    src="/images/team/sam-head.jpeg" 
                    alt="Sammy" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Sammy</h4>
              <p className="text-primary-600 mb-3">Editor & Story Consultant</p>
              <p className="text-dark-600 text-sm">
                Master of narrative structure who shapes raw footage into powerful, impactful stories.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
