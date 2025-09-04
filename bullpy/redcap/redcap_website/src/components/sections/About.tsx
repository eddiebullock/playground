import React from 'react';

const About: React.FC = () => {
  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    console.log('Image failed to load:', e.currentTarget.src);
    // Set a fallback color or placeholder
    e.currentTarget.style.backgroundColor = '#f3f4f6';
    e.currentTarget.style.display = 'flex';
    e.currentTarget.style.alignItems = 'center';
    e.currentTarget.style.justifyContent = 'center';
    e.currentTarget.style.color = '#6b7280';
    e.currentTarget.style.fontSize = '14px';
    e.currentTarget.alt = 'Image failed to load';
  };

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
            There's a gap between research and the public. Our mission is to close it — turning science into stories that everyone can get exited by.
            </p>
            <p className="text-lg text-dark-600 leading-relaxed">
              Our belief is simple: knowledge shouldn't stay locked in journals — it should reach people 
               in ways that resonate and inspire action.
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
                  src="https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/WhatsApp%20Image%202025-08-31%20at%2015.58.41%20(1).jpeg" 
                  alt="Behind the scenes - Production setup" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                  style={{height:'100%'}}
                  onError={handleImageError}
                />
                <div className="absolute inset-0 bg-black bg-opacity-20 group-hover:bg-opacity-0 transition-all duration-300" />
              </div>
            </div>
            
            {/* Bottom row - 2 images */}
            <div className="flex gap-3">
              <div className="relative group overflow-hidden rounded-lg flex-1">
                <img 
                  src="https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/WhatsApp%20Image%202025-08-30%20at%2020.53.25.jpeg" 
                  alt="Behind the scenes - Camera work" 
                  className="w-full h-24 md:h-32 object-cover transition-transform duration-300 group-hover:scale-110"
                  onError={handleImageError}
                />
                <div className="absolute inset-0 bg-black bg-opacity-20 group-hover:bg-opacity-0 transition-all duration-300" />
              </div>
              <div className="relative group overflow-hidden rounded-lg flex-1">
                <img 
                  src="https://raw.githubusercontent.com/eddiebullock/landing-page-assets/main/WhatsApp%20Image%202025-08-31%20at%2015.58.41%20(1).jpeg" 
                  alt="Behind the scenes - Team collaboration" 
                  className="w-full h-24 md:h-32 object-cover transition-transform duration-300 group-hover:scale-110"
                  onError={handleImageError}
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
                    src="https://via.placeholder.com/200x200/1f2937/ffffff?text=Eddie" 
                    alt="Eddie" 
                    className="w-full h-full object-cover"
                    onError={handleImageError}
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Eddie</h4>
              <p className="text-primary-600 mb-3">Presenter & Creative Director</p>
              <p className="text-dark-600 text-sm mb-4">
                Science presenter and PhD student at Cambridge University.
              </p>
              <a
                href="https://www.linkedin.com/in/eddie-bullock"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center w-8 h-8 bg-primary-600 hover:bg-primary-700 rounded-full transition-colors duration-200"
              >
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
            </div>

            {/* Team Member 2 - Alf */}
            <div className="text-center group">
              <div className="relative mb-6">
                <div className="w-48 h-48 mx-auto bg-gray-700 rounded-full overflow-hidden">
                  <img 
                    src="https://via.placeholder.com/200x200/1f2937/ffffff?text=Alf" 
                    alt="Alf" 
                    className="w-full h-full object-cover scale-110"
                    onError={handleImageError}
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Alf</h4>
              <p className="text-primary-600 mb-3">Director of Photography</p>
              <p className="text-dark-600 text-sm mb-4">
                Visual storyteller who transforms complex narratives into compelling cinematic experiences.
              </p>
              <a
                href="https://www.linkedin.com/in/alfie-parry-photography/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center w-8 h-8 bg-primary-600 hover:bg-primary-700 rounded-full transition-colors duration-200"
              >
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
            </div>

            {/* Team Member 3 - Sammy */}
            <div className="text-center group">
              <div className="relative mb-6">
                <div className="w-48 h-48 mx-auto bg-gray-700 rounded-full overflow-hidden">
                  <img 
                    src="https://via.placeholder.com/200x200/1f2937/ffffff?text=Sammy" 
                    alt="Sammy" 
                    className="w-full h-full object-cover"
                    onError={handleImageError}
                  />
                </div>
              </div>
              <h4 className="text-xl font-serif text-dark-900 mb-2">Sammy</h4>
              <p className="text-primary-600 mb-3">Editor & Story Consultant</p>
              <p className="text-dark-600 text-sm mb-4">
                Master of narrative structure who shapes raw footage into powerful, impactful stories.
              </p>
              <a
                href="#"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center w-8 h-8 bg-gray-400 rounded-full transition-colors duration-200 cursor-not-allowed"
                title="LinkedIn coming soon"
              >
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
