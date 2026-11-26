import { NextRequest, NextResponse } from 'next/server';

const crisisResources = [
  {
    name: 'Samaritans (UK)',
    phone: '116 123',
    url: 'https://www.samaritans.org/',
    description: '24/7 emotional support for anyone in distress.'
  },
  {
    name: 'National Suicide Prevention Lifeline (US)',
    phone: '988',
    url: 'https://988lifeline.org/',
    description: '24/7 free and confidential support for people in distress.'
  },
  {
    name: 'Crisis Text Line',
    phone: 'Text HOME to 741741',
    url: 'https://www.crisistextline.org/',
    description: '24/7 support via text for those in crisis.'
  }
];

export async function GET(req: NextRequest) {
  return NextResponse.json({ crisisResources });
} 