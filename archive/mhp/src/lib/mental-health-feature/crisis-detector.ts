const crisisKeywords = [
  'suicide', 'kill myself', 'end my life', 'self-harm', 'cut myself', 'hopeless', 'can\'t go on',
];

export function detectCrisis(input: string) {
  const found = crisisKeywords.filter(k => input.toLowerCase().includes(k));
  return {
    crisisDetected: found.length > 0,
    crisisTerms: found,
    message: found.length > 0 ?
      'We detected words indicating distress. Please consider reaching out to a crisis helpline or mental health professional.' : '',
    searchTerms: found,
    resources: found.length > 0 ? [
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
    ] : []
  };
} 