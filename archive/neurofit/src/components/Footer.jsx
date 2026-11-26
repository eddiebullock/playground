function Footer() {
  return (
    <footer 
      className="bg-calm-800 text-calm-200 py-8"
      role="contentinfo"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <p className="text-sm">
            &copy; {new Date().getFullYear()} NeuroFit. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer

