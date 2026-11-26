# NeuroFit

## Description

NeuroFit is a neurodivergent-friendly fitness platform that provides structured workouts, AI coach guidance, and low-stress routines designed specifically for autistic and neurodivergent users.

## Features

- **Landing Page**: Clean, accessible landing page with waitlist signup
- **Authentication**: Sign up, sign in, and sign out using Supabase Auth
- **Onboarding**: Personalized questionnaire to capture user preferences (sensory level, energy level, environment, etc.)
- **AI Coach**: GPT-powered chat interface that provides calm, supportive fitness guidance
- **Workout Library**: Browse and filter workouts based on user preferences
- **Progress Tracking**: Track completed workouts and view statistics

## Tech Stack

- **React 18** - UI framework
- **React Router** - Client-side routing
- **Tailwind CSS** - Styling
- **Supabase** - Authentication and database
- **OpenAI GPT** - AI Coach functionality
- **Vite** - Build tool

## Getting Started

See [SETUP.md](./SETUP.md) for detailed setup instructions.

### Quick Start

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables (copy `.env.example` to `.env` and fill in your keys)

3. Set up Supabase database (run `supabase-schema.sql` in your Supabase SQL Editor)

4. Start the development server:
```bash
npm run dev
```

## Project Structure

```
src/
├── pages/
│   ├── index.jsx          # Landing page
│   ├── auth.jsx           # Sign in/Sign up page
│   ├── dashboard.jsx      # Main dashboard
│   └── onboarding.jsx     # User preferences questionnaire
├── components/
│   ├── Header.jsx          # Navigation header
│   ├── Footer.jsx          # Footer component
│   ├── AIChat.jsx          # AI Coach chat interface
│   ├── WorkoutCard.jsx     # Individual workout card
│   ├── WorkoutList.jsx     # Workout library list
│   └── ProgressTracker.jsx # Progress tracking component
├── services/
│   ├── supabaseClient.js   # Supabase client configuration
│   └── api.js              # Database API helpers
├── utils/
│   ├── prompts.js          # GPT prompt templates
│   └── helpers.js          # Utility functions
└── styles/
    ├── index.css           # Global Tailwind styles
    └── dashboard.css       # Dashboard-specific styles
```

## Environment Variables

Create a `.env` file in the root directory:

```
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_OPENAI_API_KEY=your_openai_api_key
```

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Accessibility

This app follows WCAG 2.1 accessibility guidelines with:
- Semantic HTML
- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Screen reader compatibility

## License

[Add license information here]

