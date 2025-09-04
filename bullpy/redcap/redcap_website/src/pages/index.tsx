import Head from 'next/head'
import Hero from '@/components/sections/Hero'
import Projects from '@/components/sections/Projects'
import About from '@/components/sections/About'
import Services from '@/components/sections/Services'
import Contact from '@/components/sections/Contact'
import Footer from '@/components/layout/Footer'
import Header from '@/components/layout/Header'

export default function Home() {
  return (
    <>
      <Head>
        <title>RedCap Media - Science, Made Accessible</title>
        <meta name="description" content="We create documentaries and media that make science accessible, engaging, and understandable for everyone." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        
        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://redcapmedia.co.uk" />
        <meta property="og:title" content="RedCap Media - Science, Made Accessible" />
        <meta property="og:description" content="We create documentaries and media that make science accessible, engaging, and understandable for everyone." />
        <meta property="og:image" content="https://redcapmedia.co.uk/og-image.jpg" />
        <meta property="og:site_name" content="RedCap Media" />
        
        {/* Twitter */}
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:url" content="https://redcapmedia.co.uk" />
        <meta property="twitter:title" content="RedCap Media - Science, Made Accessible" />
        <meta property="twitter:description" content="We create documentaries and media that make science accessible, engaging, and understandable for everyone." />
        <meta property="twitter:image" content="https://redcapmedia.co.uk/og-image.jpg" />
        
        {/* Additional SEO */}
        <meta name="robots" content="index, follow" />
        <meta name="author" content="RedCap Media" />
        <meta name="keywords" content="science, documentary, media production, educational content, scientific communication, research, accessible science" />
        <link rel="canonical" href="https://redcapmedia.co.uk" />
        
        {/* Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet" />
        
        {/* Vimeo Player */}
        <script src="https://player.vimeo.com/api/player.js"></script>
      </Head>
      <div className="bg-white text-dark-900 min-h-screen">
        <Header />
        <main>
          <Hero />
          <Projects />
          <About />
          <Services />
          <Contact />
        </main>
        <Footer />
      </div>
    </>
  )
}
