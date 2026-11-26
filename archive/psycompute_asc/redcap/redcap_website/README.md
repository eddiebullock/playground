# RedCap Media - Documentary & Media Production

A cinematic landing page website for a media/documentary production company, built with Next.js and TailwindCSS.

## 🎬 About

RedCap Media creates documentaries and media that uncover hidden truths, amplify voices, and inspire change. This website showcases our projects, team, and services with a focus on authentic storytelling and social impact.

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
redcap_website/
├── src/
│   ├── components/
│   │   ├── layout/          # Header, Footer
│   │   ├── sections/        # Hero, Projects, About, Services, Contact
│   │   └── ui/              # Reusable UI components
│   ├── pages/               # Next.js pages
│   ├── styles/              # Global styles and TailwindCSS
│   ├── utils/               # Utility functions
│   └── types/               # TypeScript definitions
├── public/
│   ├── images/              # Image assets
│   └── videos/              # Video assets
└── docs/                    # Documentation
```

## 🎨 Design Features

- **Cinematic Dark Theme**: Dark backgrounds with gold/teal accents
- **Typography**: Playfair Display (serif) for headings, Inter (sans-serif) for body
- **Responsive Design**: Mobile-first approach with smooth animations
- **Accessibility**: WCAG compliant with proper contrast ratios
- **Performance**: Optimized images and lazy loading

## 📱 Sections

1. **Hero Section**: Fullscreen video background with mission statement
2. **Projects**: Grid of documentary projects with modal details
3. **About**: Mission statement and team profiles
4. **Services**: Four service offerings with process workflow
5. **Contact**: Contact form and social media links
6. **Footer**: Navigation and company information

## 🛠 Tech Stack

- **Framework**: Next.js 14
- **Styling**: TailwindCSS
- **Language**: TypeScript
- **Fonts**: Google Fonts (Playfair Display, Inter)
- **Icons**: Heroicons (SVG)
- **Deployment**: Vercel-ready

## 📸 Assets Needed

See `public/images/placeholder.txt` and `public/videos/placeholder.txt` for required assets.

## 🚀 Deployment

The project is configured for easy deployment on Vercel:

```bash
npm run build
npm run start
```

## 📝 Customization

- Update company information in component files
- Replace placeholder images and videos
- Modify color scheme in `tailwind.config.js`
- Add new sections as needed

## 📄 License

© 2024 RedCap Media. All rights reserved.
