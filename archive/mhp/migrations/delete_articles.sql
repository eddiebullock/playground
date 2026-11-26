-- Delete articles from psychology and neurodiversity categories
DELETE FROM articles 
WHERE category IN ('psychology', 'neurodiversity')
AND slug IN (
  'emotion-and-affect',
  'cognitive-development',
  'memory-and-learning',
  'developmental-psychology',
  'brain-psychopathology',
  'neurodiversity-overview',
  'difference-vs-disorder',
  'autism',
  'adhd',
  'dyslexia',
  'dyspraxia',å
  'tourette-syndrome',
  'stigma-and-neurodiversity',
  'neurodiversity-workplace-inclusion'
); 