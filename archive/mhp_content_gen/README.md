# Mental Health & Psychology Content Generator

A proprietary CLI tool for generating structured article content about mental health, psychology, neuroscience, and related topics using GPT-4.

## License

This software is proprietary and confidential. All rights are reserved by Eddie Bullock.

Copyright (c) 2024 Eddie Bullock. All Rights Reserved.

No part of this software, including but not limited to the source code, documentation, 
specifications, and design, may be reproduced, distributed, or transmitted in any form 
or by any means without the prior written permission of the copyright holder.

## Features

- Generate structured article content in JSON format
- Supports multiple article categories (mental health, neuroscience, psychology, etc.)
- Validates generated content against a strict schema
- Easy integration with Supabase for content storage
- Command-line interface for easy use
- Generate multiple articles in sequence
- Bulk upload to database

## Setup

1. Clone the repository:
```bash
git clone https://github.com/eddiebullock/mhp_content_gen.git
cd mhp_content_gen
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## Usage

Generate a single article:
```bash
npm start -- --topic "Anxiety Disorders" --category "mental_health" --model "gpt-4o-mini"
```

Generate multiple articles:
```bash
npm run generate-multiple -- -t "Anxiety Disorders,Depression,OCD" -c "mental_health" -m "gpt-4o-mini"
```

Update specific sections of existing articles:
```bash
# Update summaries of the last 10 articles
npm run update-content -- -s summary -c 10

# Update overviews of the last 5 articles
npm run update-content -- -s overview -c 5

# Update practical takeaways of the last 3 articles
npm run update-content -- -s practical_takeaways -c 3
```

Upload articles to database:
```bash
npm run bulk-upload
```

Available categories:
- mental_health
- neuroscience
- psychology
- brain_health
- neurodiversity
- interventions
- lifestyle_factors
- lab_testing
- risk_factors

## Output

The tool generates a JSON file (`articles-data.json`) that matches the required schema for the Supabase database. The generated content includes:

### Base Fields (All Categories)
- `title`: Article title
- `slug`: URL-friendly version of the title
- `summary`: Brief overview (2-3 sentences)
- `category`: Article category (mental_health, neuroscience, psychology, brain_health, neurodiversity, interventions, lifestyle_factors, lab_testing)
- `overview`: Comprehensive introduction
- `evidence_summary`: Consolidated evidence from key_evidence, effectiveness, and evidence_base fields
- `practical_applications`: Consolidated practical information from practical_takeaways and practical_applications fields
- `future_directions`: Emerging research and future developments
- `references_and_resources`: Key references and further reading
- `status`: Article status (published, draft, archived)
- `tags`: Array of relevant tags

### Category-Specific Fields

#### Mental Health Articles
- `prevalence`: Prevalence statistics and demographic information
- `causes_and_mechanisms`: Biological and environmental causes
- `symptoms_and_impact`: Symptoms and their impact on daily life
- `common_myths`: Common misconceptions and myths

#### Neuroscience/Psychology/Brain Health Articles
- `definition`: Clear definition of the concept
- `mechanisms`: Underlying mechanisms and processes
- `relevance`: Relevance to everyday life
- `key_studies`: Important research studies
- `common_misconceptions`: Common misconceptions

#### Neurodiversity Articles
- `neurodiversity_perspective`: Neurodiversity paradigm and perspective
- `common_strengths_and_challenges`: Strengths and challenges
- `prevalence_and_demographics`: Prevalence and demographic information
- `mechanisms_and_understanding`: Mechanisms and understanding
- `common_misconceptions`: Common misconceptions
- `lived_experience`: Lived experience perspectives

#### Intervention/Lifestyle Articles
- `how_it_works`: How the intervention or lifestyle factor works
- `common_myths`: Common misconceptions
- `risks_and_limitations`: Potential risks and limitations
- `reliability_score`: Reliability score (0-1) based on effect sizes and replication frequency

#### Lab Testing Articles
- `how_it_works`: How the test works
- `applications`: Applications and uses
- `strengths_and_limitations`: Strengths and limitations
- `risks_and_limitations`: Potential risks and limitations

#### Risk Factors Articles
- `overview`: What the risk factor is
- `prevalence`: How common it is
- `mechanisms`: How it affects mental health
- `evidence_summary`: Research backing
- `modifiable_factors`: What can be changed
- `protective_factors`: What reduces risk
- `practical_takeaways`: Key points for users

### Schema Changes (Latest Update)

The schema has been updated to consolidate related fields and add reliability scoring:

1. **Consolidated Evidence Fields**: `key_evidence`, `effectiveness`, and `evidence_base` are now combined into a single `evidence_summary` field
2. **Consolidated Practical Fields**: `practical_takeaways` and `practical_applications` are now combined into a single `practical_applications` field
3. **Reliability Scores**: Intervention and lifestyle articles now include a `reliability_score` (0-1) based on:
   - Effect sizes (0.1-0.3 = small, 0.3-0.5 = medium, >0.5 = large)
   - Number of studies/replications
   - Quality of evidence (RCTs, meta-analyses, etc.)
   - Consistency of findings across studies

## Validation

The tool includes built-in validation to ensure generated content matches the required schema. Run validation:

```bash
npm test
```

## Support

For licensing inquiries or support, please contact the copyright holder.

Reliability Score = (Effect Size × 0.4) + (Study Quality × 0.3) + (Replication × 0.15) + (Sample Size × 0.15)


