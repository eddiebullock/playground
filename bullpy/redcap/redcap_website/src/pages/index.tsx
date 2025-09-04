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
        <title>RedCap Media - Stories That Change Perspectives</title>
        <meta name="description" content="We create documentaries and media that uncover hidden truths, amplify voices, and inspire change." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet" />
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
