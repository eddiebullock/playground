import { Command } from 'commander';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { validateArticle, getSchemaForCategory } from './schema.js';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config();

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize CLI
const program = new Command();

program
  .name('mhp-content-gen')
  .description('Generate mental health and psychology article content using GPT')
  .version('1.0.0')
  .requiredOption('-t, --topic <string>', 'The topic of the article')
  .requiredOption('-c, --category <string>', 'The category of the article')
  .option('-o, --output <string>', 'Output file path', 'articles-data.json')
  .option('-m, --model <string>', 'GPT model to use (ignored, always uses gpt-4o-mini)', 'gpt-4o-mini')
  .option('--no-clear', 'Skip clearing the output file (used when called from generate-multiple.js)')
  .parse(process.argv);

const options = program.opts();
const model = 'gpt-4o-mini'; // Always use gpt-4o-mini regardless of user input

// Helper function to generate slug from title
function generateSlug(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

// Example article structure for GPT reference
const exampleArticle = {
  title: "Understanding Autism Spectrum Disorder",
  slug: "understanding-autism-spectrum-disorder",
  summary: "A comprehensive overview of Autism Spectrum Disorder (ASD), including its neurobiological basis, diagnostic criteria, and evidence-based interventions. This article examines current research on ASD's prevalence, mechanisms, and practical approaches to support.",
  category: "neurodiversity",
  overview: "Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by differences in social communication and interaction, alongside restricted and repetitive patterns of behavior [1]. Research indicates a prevalence of approximately 1-2% of the population [2], with increasing recognition of the condition's diverse presentation across genders and ages.",
  neurodiversity_perspective: "The neurodiversity paradigm views ASD as a natural variation in human neurology rather than a disorder requiring 'fixing' [3]. This perspective emphasizes the importance of understanding and accommodating neurological differences while recognizing both challenges and strengths associated with ASD.",
  common_strengths_and_challenges: "Individuals with ASD often demonstrate exceptional attention to detail, pattern recognition, and deep focus in areas of interest [4]. Common challenges may include sensory processing differences, social communication variations, and executive functioning differences. Research suggests these traits exist on a spectrum, with significant individual variation [5].",
  prevalence_and_demographics: "Recent epidemiological studies estimate ASD prevalence at 1-2% of the population [2]. Research indicates potential underdiagnosis in females and older adults, with growing recognition of the condition's presentation across different genders and age groups [6].",
  mechanisms_and_understanding: "Current research suggests ASD involves complex interactions between genetic and environmental factors [7]. Neuroimaging studies indicate differences in brain connectivity and structure, particularly in regions associated with social processing and sensory integration [8].",
  evidence_summary: "Meta-analyses of intervention studies suggest that early behavioral interventions can improve outcomes in communication and adaptive skills [9]. However, research emphasizes the importance of individualized approaches that respect neurodiversity principles [10].",
  common_misconceptions: "Common misconceptions include viewing ASD as a childhood condition that can be 'outgrown,' or assuming all individuals with ASD have exceptional savant abilities. Research indicates ASD is a lifelong condition with diverse presentations [11].",
  practical_takeaways: "Effective support strategies include: creating sensory-friendly environments, using clear communication methods, and implementing structured routines. Research supports the effectiveness of these approaches when tailored to individual needs [12].",
  lived_experience: "Autistic individuals report that understanding and acceptance from others significantly impacts their well-being [13]. Research emphasizes the importance of including autistic voices in research and support planning [14].",
  future_directions: "Emerging research focuses on understanding the diverse presentations of ASD across the lifespan, developing more effective support strategies, and improving access to diagnosis and services [15].",
  references_and_resources: "[1] American Psychiatric Association. (2022). Diagnostic and Statistical Manual of Mental Disorders (5th ed., text rev.)...",
  status: "draft",
  tags: ["autism", "neurodiversity", "neurodevelopment", "sensory processing", "social communication"],
  key_evidence: "Research indicates that ASD is associated with distinct patterns of brain connectivity and neural processing [8]. Studies have shown that early intervention can lead to significant improvements in communication and social skills [9]. Neuroimaging research has identified differences in brain structure and function, particularly in regions associated with social cognition and sensory processing [7]."
};

// Example psychology article
const examplePsychologyArticle = {
  title: "Understanding Cognitive Development",
  slug: "understanding-cognitive-development",
  summary: "A comprehensive exploration of cognitive development theories and research, examining how thinking, learning, and problem-solving abilities evolve across the lifespan. This article synthesizes key theories and evidence-based research on cognitive development.",
  category: "psychology",
  overview: "Cognitive development refers to the growth and change in intellectual abilities across the lifespan [1]. Research indicates that cognitive development follows both universal patterns and individual variations, influenced by biological, environmental, and social factors [2].",
  definition: "Cognitive development is the process through which humans acquire, organize, and use knowledge and skills. It encompasses changes in thinking, reasoning, problem-solving, and information processing abilities that occur from infancy through adulthood [3].",
  core_principles: "Key principles of cognitive development include: the interaction of nature and nurture, the importance of early experiences, and the role of social interaction in learning [4]. Research demonstrates that cognitive development is both continuous and stage-like, with critical periods for certain types of learning [5].",
  relevance: "Understanding cognitive development is crucial for education, parenting, and mental health practice [6]. Research shows that developmentally appropriate interventions can significantly impact learning outcomes and cognitive abilities [7].",
  key_studies_and_theories: "Piaget's theory of cognitive development [8] and Vygotsky's sociocultural theory [9] provide foundational frameworks. Recent neuroimaging studies have enhanced our understanding of the neural basis of cognitive development [10].",
  common_misconceptions: "Common misconceptions include viewing cognitive development as purely biological or assuming all children develop at the same rate. Research indicates significant individual variation in developmental trajectories [11].",
  practical_applications: "Practical applications include: developmentally appropriate educational strategies, early intervention programs, and cognitive assessment tools. Research supports the effectiveness of these approaches when tailored to individual needs [12].",
  future_directions: "Emerging research focuses on the role of technology in cognitive development, the impact of environmental factors, and the development of more precise assessment tools [13].",
  references_and_resources: "[1] Piaget, J. (1952). The origins of intelligence in children...",
  status: "draft",
  tags: ["cognitive development", "psychology", "learning", "education", "child development"],
  key_evidence: "Longitudinal studies demonstrate that early cognitive stimulation significantly impacts later academic achievement [14]. Neuroimaging research has identified critical periods for different types of learning [15]. Meta-analyses of intervention studies show that developmentally appropriate programs can enhance cognitive abilities [16].",
  practical_takeaways: "Effective approaches to supporting cognitive development include: providing rich learning environments, engaging in interactive activities, and using developmentally appropriate teaching methods. Research emphasizes the importance of individual differences and the role of social interaction in cognitive growth."
};

// Example risk factors article
const exampleRiskFactorsArticle = {
  title: "Understanding Childhood Trauma as a Risk Factor",
  slug: "understanding-childhood-trauma-risk-factor",
  summary: "A comprehensive examination of childhood trauma as a significant risk factor for mental health conditions, including its prevalence, biological mechanisms, and evidence-based approaches to risk reduction and protective factors.",
  category: "risk_factors",
  overview: "Childhood trauma encompasses experiences of abuse, neglect, violence, or other adverse events that occur during critical developmental periods [1]. Research indicates that childhood trauma is one of the most significant risk factors for developing mental health conditions later in life, with effects that can persist across the lifespan [2].",
  prevalence: "Studies estimate that approximately 60% of adults report experiencing at least one adverse childhood experience (ACE), with 25% reporting multiple ACEs [3]. The prevalence varies by demographic factors, with higher rates among certain populations and socioeconomic groups [4]. Research suggests that childhood trauma affects individuals across all backgrounds, though some groups may be more vulnerable due to systemic factors [5].",
  mechanisms: "Childhood trauma affects mental health through multiple biological and psychological mechanisms. Neurobiological research shows that trauma can alter brain development, particularly in regions responsible for stress regulation, emotional processing, and executive function [6]. The hypothalamic-pituitary-adrenal (HPA) axis, which regulates stress responses, may become dysregulated, leading to increased vulnerability to stress-related conditions [7]. Additionally, trauma can affect the development of neural circuits involved in emotion regulation, social cognition, and threat detection [8].",
  evidence_summary: "Meta-analyses of longitudinal studies demonstrate that childhood trauma significantly increases the risk of developing depression, anxiety, post-traumatic stress disorder, and other mental health conditions [9]. Research indicates a dose-response relationship, where greater exposure to trauma correlates with increased risk [10]. Studies have also identified specific biological markers associated with childhood trauma, including altered cortisol patterns, changes in brain structure and function, and epigenetic modifications [11]. The evidence base includes large-scale epidemiological studies, neuroimaging research, and intervention studies that demonstrate the long-term impact of early trauma [12].",
  modifiable_factors: "Several factors can modify the impact of childhood trauma on mental health outcomes. Early intervention and trauma-informed care can significantly reduce the long-term effects of trauma [13]. Access to mental health services, particularly trauma-focused therapies like cognitive behavioral therapy and eye movement desensitization and reprocessing (EMDR), has been shown to be effective [14]. Building strong support networks, developing healthy coping mechanisms, and creating safe environments can help mitigate trauma's effects [15]. Additionally, addressing systemic factors such as poverty, discrimination, and lack of access to resources can reduce trauma exposure and improve outcomes [16].",
  protective_factors: "Research identifies several protective factors that can buffer against the negative effects of childhood trauma. Strong, supportive relationships with caregivers, teachers, or mentors can provide emotional security and promote resilience [17]. Access to mental health services and early intervention programs can help address trauma-related symptoms before they become chronic [18]. Developing healthy coping strategies, such as mindfulness, exercise, and creative expression, can enhance emotional regulation and stress management [19]. Additionally, factors such as high self-esteem, optimism, and a sense of purpose can contribute to resilience in the face of adversity [20].",
  practical_takeaways: "Understanding childhood trauma as a risk factor requires a comprehensive approach that addresses both prevention and intervention. Key strategies include: recognizing the signs of trauma in children and adults, providing trauma-informed care and support, advocating for policies that reduce trauma exposure, and promoting access to mental health services. It's important to remember that while childhood trauma is a significant risk factor, it doesn't determine outcomes - many individuals demonstrate remarkable resilience and recovery. Early identification and intervention can significantly improve long-term mental health outcomes, and building supportive environments can help prevent trauma exposure in the first place.",
  reliability_score: 0.85,
  references: "[1] Felitti, V. J., et al. (1998). Relationship of childhood abuse and household dysfunction to many of the leading causes of death in adults. American Journal of Preventive Medicine, 14(4), 245-258. [2] Teicher, M. H., & Samson, J. A. (2016). Annual research review: Enduring neurobiological effects of childhood abuse and neglect. Journal of Child Psychology and Psychiatry, 57(3), 241-266. [3] Merrick, M. T., et al. (2018). Prevalence of adverse childhood experiences from the 2011-2014 Behavioral Risk Factor Surveillance System in 23 states. JAMA Pediatrics, 172(11), 1038-1044. [4] Hughes, K., et al. (2017). The effect of multiple adverse childhood experiences on health: a systematic review and meta-analysis. The Lancet Public Health, 2(8), e356-e366. [5] Anda, R. F., et al. (2006). The enduring effects of abuse and related adverse experiences in childhood. European Archives of Psychiatry and Clinical Neuroscience, 256(3), 174-186. [6] Teicher, M. H., et al. (2003). The neurobiological consequences of early stress and childhood maltreatment. Neuroscience & Biobehavioral Reviews, 27(1-2), 33-44. [7] Heim, C., & Nemeroff, C. B. (2001). The role of childhood trauma in the neurobiology of mood and anxiety disorders: preclinical and clinical studies. Biological Psychiatry, 49(12), 1023-1039. [8] McCrory, E. J., et al. (2010). Research review: The neurobiology and genetics of maltreatment and adversity. Journal of Child Psychology and Psychiatry, 51(10), 1079-1095. [9] Norman, R. E., et al. (2012). The long-term health consequences of child physical abuse, emotional abuse, and neglect: a systematic review and meta-analysis. PLoS Medicine, 9(11), e1001349. [10] Anda, R. F., et al. (2006). The enduring effects of abuse and related adverse experiences in childhood. European Archives of Psychiatry and Clinical Neuroscience, 256(3), 174-186. [11] Teicher, M. H., & Samson, J. A. (2016). Annual research review: Enduring neurobiological effects of childhood abuse and neglect. Journal of Child Psychology and Psychiatry, 57(3), 241-266. [12] McLaughlin, K. A., et al. (2012). Childhood adversities and first onset of psychiatric disorders in a national sample of US adolescents. Archives of General Psychiatry, 69(11), 1151-1160. [13] Cohen, J. A., et al. (2012). Trauma-focused CBT for children with co-occurring trauma and behavior problems. Child Abuse & Neglect, 36(4), 321-332. [14] Shapiro, F. (2014). The role of eye movement desensitization and reprocessing (EMDR) therapy in medicine: addressing the psychological and physical symptoms stemming from adverse life experiences. The Permanente Journal, 18(1), 71-77. [15] Masten, A. S. (2001). Ordinary magic: Resilience processes in development. American Psychologist, 56(3), 227-238. [16] Shonkoff, J. P., et al. (2012). The lifelong effects of early childhood adversity and toxic stress. Pediatrics, 129(1), e232-e246. [17] Werner, E. E. (2013). What can we learn about resilience from large-scale longitudinal studies? In S. Goldstein & R. B. Brooks (Eds.), Handbook of resilience in children (pp. 87-102). Springer. [18] Cohen, J. A., & Mannarino, A. P. (2008). Trauma-focused cognitive behavioural therapy for children and parents. Child and Adolescent Mental Health, 13(4), 158-162. [19] Southwick, S. M., & Charney, D. S. (2012). The science of resilience: implications for the prevention and treatment of depression. Science, 338(6103), 79-82. [20] Masten, A. S., & Narayan, A. J. (2012). Child development in the context of disaster, war, and terrorism: Pathways of risk and resilience. Annual Review of Psychology, 63, 227-257. [21] van der Kolk, B. A. (2014). The body keeps the score: Brain, mind, and body in the healing of trauma. Viking. [22] Bethell, C. D., et al. (2017). Methods to assess adverse childhood experiences of children and families: toward approaches to promote child well-being in policy and practice. Academic Pediatrics, 17(7), S51-S69.",
  future_directions: "Emerging research focuses on developing more effective early intervention strategies, understanding individual differences in trauma response, and identifying novel therapeutic approaches. Studies are exploring the role of neuroplasticity in trauma recovery, the potential of pharmacological interventions, and the development of personalized treatment approaches [21]. Research is also examining the impact of systemic factors and developing community-based prevention programs [22].",
  references_and_resources: "[1] Felitti, V. J., et al. (1998). Relationship of childhood abuse and household dysfunction to many of the leading causes of death in adults. American Journal of Preventive Medicine, 14(4), 245-258. [2] Teicher, M. H., & Samson, J. A. (2016). Annual research review: Enduring neurobiological effects of childhood abuse and neglect. Journal of Child Psychology and Psychiatry, 57(3), 241-266. [3] Merrick, M. T., et al. (2018). Prevalence of adverse childhood experiences from the 2011-2014 Behavioral Risk Factor Surveillance System in 23 states. JAMA Pediatrics, 172(11), 1038-1044. [4] Hughes, K., et al. (2017). The effect of multiple adverse childhood experiences on health: a systematic review and meta-analysis. The Lancet Public Health, 2(8), e356-e366. [5] Anda, R. F., et al. (2006). The enduring effects of abuse and related adverse experiences in childhood. European Archives of Psychiatry and Clinical Neuroscience, 256(3), 174-186. [6] Teicher, M. H., et al. (2003). The neurobiological consequences of early stress and childhood maltreatment. Neuroscience & Biobehavioral Reviews, 27(1-2), 33-44. [7] Heim, C., & Nemeroff, C. B. (2001). The role of childhood trauma in the neurobiology of mood and anxiety disorders: preclinical and clinical studies. Biological Psychiatry, 49(12), 1023-1039. [8] McCrory, E. J., et al. (2010). Research review: The neurobiology and genetics of maltreatment and adversity. Journal of Child Psychology and Psychiatry, 51(10), 1079-1095. [9] Norman, R. E., et al. (2012). The long-term health consequences of child physical abuse, emotional abuse, and neglect: a systematic review and meta-analysis. PLoS Medicine, 9(11), e1001349. [10] Anda, R. F., et al. (2006). The enduring effects of abuse and related adverse experiences in childhood. European Archives of Psychiatry and Clinical Neuroscience, 256(3), 174-186. [11] Teicher, M. H., & Samson, J. A. (2016). Annual research review: Enduring neurobiological effects of childhood abuse and neglect. Journal of Child Psychology and Psychiatry, 57(3), 241-266. [12] McLaughlin, K. A., et al. (2012). Childhood adversities and first onset of psychiatric disorders in a national sample of US adolescents. Archives of General Psychiatry, 69(11), 1151-1160. [13] Cohen, J. A., et al. (2012). Trauma-focused CBT for children with co-occurring trauma and behavior problems. Child Abuse & Neglect, 36(4), 321-332. [14] Shapiro, F. (2014). The role of eye movement desensitization and reprocessing (EMDR) therapy in medicine: addressing the psychological and physical symptoms stemming from adverse life experiences. The Permanente Journal, 18(1), 71-77. [15] Masten, A. S. (2001). Ordinary magic: Resilience processes in development. American Psychologist, 56(3), 227-238. [16] Shonkoff, J. P., et al. (2012). The lifelong effects of early childhood adversity and toxic stress. Pediatrics, 129(1), e232-e246. [17] Werner, E. E. (2013). What can we learn about resilience from large-scale longitudinal studies? In S. Goldstein & R. B. Brooks (Eds.), Handbook of resilience in children (pp. 87-102). Springer. [18] Cohen, J. A., & Mannarino, A. P. (2008). Trauma-focused cognitive behavioural therapy for children and parents. Child and Adolescent Mental Health, 13(4), 158-162. [19] Southwick, S. M., & Charney, D. S. (2012). The science of resilience: implications for the prevention and treatment of depression. Science, 338(6103), 79-82. [20] Masten, A. S., & Narayan, A. J. (2012). Child development in the context of disaster, war, and terrorism: Pathways of risk and resilience. Annual Review of Psychology, 63, 227-257. [21] van der Kolk, B. A. (2014). The body keeps the score: Brain, mind, and body in the healing of trauma. Viking. [22] Bethell, C. D., et al. (2017). Methods to assess adverse childhood experiences of children and families: toward approaches to promote child well-being in policy and practice. Academic Pediatrics, 17(7), S51-S69.",
  status: "draft",
  tags: ["childhood trauma", "risk factors", "mental health", "adverse childhood experiences", "resilience", "trauma-informed care"],
  key_evidence: "Meta-analyses demonstrate that childhood trauma increases the risk of mental health conditions by 2-4 times [9]. Neuroimaging studies show structural and functional brain changes in trauma-exposed individuals [6]. Longitudinal studies reveal dose-response relationships between trauma exposure and mental health outcomes [10]. Intervention research shows that trauma-focused therapies can significantly improve outcomes [13].",
  practical_takeaways: "Key strategies for addressing childhood trauma include: early identification and intervention, trauma-informed care approaches, building supportive relationships, developing healthy coping mechanisms, and advocating for systemic changes that reduce trauma exposure. Remember that resilience is possible and many individuals demonstrate remarkable recovery with appropriate support."
};

// Helper function to process GPT response
function processArticleResponse(article) {
  console.log(chalk.yellow('\nDebug: Original article format:'));
  console.log(chalk.yellow('keyEvidence/key_evidence type:', typeof (article.keyEvidence || article.key_evidence), Array.isArray(article.keyEvidence || article.key_evidence) ? '(array)' : ''));
  console.log(chalk.yellow('practicalTakeaways/practical_takeaways type:', typeof (article.practicalTakeaways || article.practical_takeaways), Array.isArray(article.practicalTakeaways || article.practical_takeaways) ? '(array)' : ''));

  // Handle field name conversion and array to string conversion
  const processed = { ...article };
  
  // Convert camelCase to snake_case and handle arrays
  if (processed.keyEvidence !== undefined) {
    processed.key_evidence = Array.isArray(processed.keyEvidence) 
      ? processed.keyEvidence.join('. ') + '.'
      : processed.keyEvidence;
    delete processed.keyEvidence;
  }
  
  if (processed.practicalTakeaways !== undefined) {
    processed.practical_takeaways = Array.isArray(processed.practicalTakeaways)
      ? processed.practicalTakeaways.join('. ') + '.'
      : processed.practicalTakeaways;
    delete processed.practicalTakeaways;
  }

  console.log(chalk.yellow('\nDebug: After processing:'));
  console.log(chalk.yellow('key_evidence type:', typeof processed.key_evidence));
  console.log(chalk.yellow('practical_takeaways type:', typeof processed.practical_takeaways));

  return processed;
}

// Helper function to get example article for category
function getExampleArticle(category) {
  switch (category) {
    case 'psychology':
      return examplePsychologyArticle;
    case 'neurodiversity':
      return exampleArticle;
    case 'risk_factors':
      return exampleRiskFactorsArticle;
    default:
      return exampleArticle; // Default to neurodiversity example
  }
}

// Helper function to generate GPT prompt based on category
function generatePrompt(topic, category) {
  const exampleArticle = getExampleArticle(category);
  const schema = getSchemaForCategory(category);
  
  // Get required fields for the category
  const requiredFields = Object.entries(schema.shape)
    .filter(([_, field]) => !field.isOptional())
    .map(([name]) => name);

  const basePrompt = `
Generate a comprehensive, evidence-based article about "${topic}" in the context of ${category}.

# Article Requirements
- Write in a clear, accessible style for a general audience
- Use scientific accuracy with Vancouver citation style [1]
- Include practical, actionable insights
- Address common misconceptions
- Provide evidence-based recommendations
- Use everyday examples and analogies
- Avoid unnecessary jargon - explain technical terms

# Category-Specific Requirements
${category} Articles
The following fields are REQUIRED and must be included:
${requiredFields.map(field => `- ${field}`).join('\n')}

## Important Format Requirements
- Use snake_case for all field names (e.g., practical_applications, evidence_summary)
- ALL text fields must be returned as strings, NOT arrays
- For evidence_summary, combine key_evidence, effectiveness, and evidence_base into a single cohesive paragraph
- For practical_applications, combine practical_takeaways and practical_applications into a single cohesive paragraph
- Use proper paragraph formatting with complete sentences
- Include transitions between ideas
- Maintain consistent citation style throughout
- For intervention and lifestyle categories, include a reliability_score (0-1) based on effect sizes and replication frequency

## Category-Specific Instructions
${getCategorySpecificInstructions(category)}

# Required Schema
The article must be formatted as a JSON object matching this schema:
${JSON.stringify(schema.shape, null, 2)}

# Example Output
Here's an example of a well-structured article in the ${category} category:
${JSON.stringify(exampleArticle, null, 2)}

# Important Notes
1. Follow the exact schema structure
2. Use snake_case for all field names
3. Include ALL required fields listed above
4. Include numerical citations in square brackets [1]
5. Provide detailed references at the end
6. Ensure all text fields are comprehensive and well-supported
7. Write in clear, accessible language while maintaining scientific accuracy
8. ALL text fields must be strings, not arrays
9. Combine multiple points into cohesive paragraphs
10. Use everyday examples and analogies to explain complex concepts
11. Avoid unnecessary jargon - explain technical terms when used
12. Write in a conversational, engaging style that's easy to understand
13. For intervention and lifestyle categories, calculate reliability_score based on:
    - Effect sizes (0.1-0.3 = small, 0.3-0.5 = medium, >0.5 = large)
    - Number of studies/replications
    - Quality of evidence (RCTs, meta-analyses, etc.)
    - Consistency of findings across studies`;

  return basePrompt;
}

// Helper function to get category-specific instructions
function getCategorySpecificInstructions(category) {
  switch (category) {
    case 'mental_health':
      return `
## Mental Health Articles
Focus on understanding, prevalence, causes, symptoms, and evidence-based approaches.
Required fields: overview, prevalence, causes_and_mechanisms, symptoms_and_impact, evidence_summary, practical_applications, common_myths, future_directions, references_and_resources
- Provide clear definitions and prevalence statistics
- Explain biological and environmental causes
- Describe symptoms and their impact on daily life
- Include evidence-based treatment approaches
- Address common misconceptions
- Offer practical coping strategies`;

    case 'neuroscience':
      return `
## Neuroscience Articles
Focus on brain mechanisms, research findings, and scientific understanding.
Required fields: overview, definition, mechanisms, relevance, key_studies, evidence_summary, practical_applications, common_misconceptions, future_directions, references_and_resources
- Explain brain mechanisms clearly
- Highlight key research studies
- Connect neuroscience to everyday life
- Address common misconceptions
- Provide practical implications`;

    case 'psychology':
      return `
## Psychology Articles
Focus on psychological principles, theories, and applications.
Required fields: overview, definition, mechanisms, relevance, key_studies, evidence_summary, practical_applications, common_misconceptions, future_directions, references_and_resources
- Explain psychological concepts clearly
- Include key theories and research
- Show practical applications
- Address misconceptions
- Provide evidence-based insights`;

    case 'brain_health':
      return `
## Brain Health Articles
Focus on maintaining and optimizing brain function.
Required fields: overview, definition, mechanisms, relevance, key_studies, evidence_summary, practical_applications, common_misconceptions, future_directions, references_and_resources
- Explain brain health concepts
- Include evidence-based strategies
- Provide practical tips
- Address common myths
- Focus on prevention and optimization`;

    case 'neurodiversity':
      return `
## Neurodiversity Articles
Focus on neurodivergent perspectives, strengths, and support.
Required fields: overview, neurodiversity_perspective, common_strengths_and_challenges, prevalence_and_demographics, mechanisms_and_understanding, evidence_summary, practical_applications, common_misconceptions, lived_experience, future_directions, references_and_resources
- Emphasize neurodiversity as natural variation
- Highlight strengths and challenges
- Include lived experience perspectives
- Provide evidence-based support strategies
- Address misconceptions respectfully
- Focus on acceptance and accommodation`;

    case 'interventions':
      return `
## Intervention Articles
Focus on evidence-based interventions and their effectiveness.
Required fields: overview, how_it_works, evidence_summary, practical_applications, common_myths, risks_and_limitations, reliability_score, future_directions, references_and_resources
- Explain how the intervention works
- Provide comprehensive evidence summary
- Include practical application guidelines
- Address risks and limitations
- Include reliability score (0-1) based on effect sizes and replication
- Focus on evidence-based approaches`;

    case 'lifestyle_factors':
      return `
## Lifestyle Factors Articles
Focus on lifestyle choices that impact mental health and brain function.
Required fields: overview, how_it_works, evidence_summary, practical_applications, common_myths, risks_and_limitations, reliability_score, future_directions, references_and_resources
- Explain the lifestyle factor's impact
- Provide evidence-based recommendations
- Include practical implementation tips
- Address common misconceptions
- Include reliability score (0-1) based on effect sizes and replication
- Focus on sustainable changes`;

    case 'lab_testing':
      return `
## Lab Testing Articles
Focus on laboratory tests and their applications in mental health.
Required fields: overview, how_it_works, applications, evidence_summary, practical_applications, strengths_and_limitations, risks_and_limitations, future_directions, references_and_resources
- Explain how the test works
- Describe applications and uses
- Include evidence for effectiveness
- Address limitations and risks
- Provide practical guidance`;

    case 'risk_factors':
      return `
## Risk Factors Articles
Focus on factors that increase risk for mental health conditions and how to address them.
Required fields: overview, prevalence, mechanisms, evidence_summary, modifiable_factors, protective_factors, practical_takeaways, reliability_score, future_directions, references_and_resources

### Content Block Structure (MUST follow this exact structure):
1. **overview** - What the risk factor is
2. **prevalence** - How common it is
3. **mechanisms** - How it affects mental health
4. **evidence_summary** - Research backing
5. **modifiable_factors** - What can be changed
6. **protective_factors** - What reduces risk
7. **practical_takeaways** - Key points for users
8. **reliability_score** - Reliability score (0-1) based on effect sizes and replication frequency

- Explain the risk factor clearly and comprehensively
- Provide accurate prevalence statistics
- Describe biological and psychological mechanisms
- Include robust research evidence
- Focus on modifiable factors that can be addressed
- Highlight protective factors and resilience
- Provide actionable, evidence-based recommendations
- Use clear, accessible language while maintaining scientific accuracy
- Include practical strategies for risk reduction
- Address common misconceptions about the risk factor
- Include proper academic references in Vancouver citation style
- Calculate reliability_score (0-1) based on:
  - Effect sizes (0.1-0.3 = small, 0.3-0.5 = medium, >0.5 = large)
  - Number of studies/replications
  - Quality of evidence (RCTs, meta-analyses, etc.)
  - Consistency of findings across studies`;

    default:
      return '';
  }
}

// Helper function to nest content fields under content_blocks
function nestContentBlocks(article) {
  // List of fields that should go inside content_blocks
  const contentFields = [
    'overview',
    'prevalence',
    'causes_and_mechanisms',
    'symptoms_and_impact',
    'evidence_summary',
    'common_myths',
    'definition',
    'mechanisms',
    'relevance',
    'key_studies',
    'key_studies_and_theories',
    'core_principles',
    'practical_applications',
    'practical_implications',
    'common_misconceptions',
    'neurodiversity_perspective',
    'common_strengths_and_challenges',
    'prevalence_and_demographics',
    'mechanisms_and_understanding',
    'lived_experience',
    'how_it_works',
    'evidence_base',
    'effectiveness',
    'practical_applications',
    'risks_and_limitations',
    'future_directions',
    'references_and_resources'
  ];
  const content_blocks = {};
  const topLevel = {};
  for (const key in article) {
    if (contentFields.includes(key)) {
      content_blocks[key] = article[key];
    } else {
      topLevel[key] = article[key];
    }
  }
  topLevel.content_blocks = content_blocks;
  return topLevel;
}

// Main article generation function
async function generateArticle(topic, category, model) {
  console.log(chalk.blue(`\nGenerating article about "${topic}" in category "${category}"...`));

  // Get the correct schema for the category
  const schema = getSchemaForCategory(category);
  const requiredFields = Object.entries(schema.shape)
    .filter(([_, field]) => !field.isOptional())
    .map(([name]) => name);

  // Get category-specific instructions
  const categorySpecificPrompt = getCategorySpecificInstructions(category);

  // Get section order based on category
  let sectionOrder = '';
  if (category === 'neurodiversity') {
    sectionOrder = `
      Structure the article in this order:
      1. Introduction (overview)
      2. Neurodiversity Perspective (required)
      3. Common Strengths and Challenges (required)
      4. Prevalence and Demographics (required)
      5. Mechanisms and Understanding (required)
      6. Evidence Summary
      7. Common Misconceptions (required)
      8. Practical Takeaways
      9. Lived Experience (required)
      10. References and Resources (required)
    `;
  } else if (category === 'mental_health') {
    sectionOrder = `
      Structure the article in this order:
      1. Introduction (overview)
      2. Evidence Summary
      3. Practical Takeaways
      4. References and Resources (required)
    `;
  } else if (category === 'interventions') {
    sectionOrder = `
      Structure the article in this order:
      1. Introduction (overview)
      2. Evidence Summary
      3. Practical Takeaways
      4. Future Directions
      5. References and Resources (required)
    `;
  }

  const prompt = `Generate a comprehensive article about ${topic} in the context of ${category}. 
    ${categorySpecificPrompt}
    ${sectionOrder}
    
    The article should be informative, evidence-based, and accessible to a general audience.
    Include practical takeaways and key evidence from research.

    # Writing Guidelines
    ## Summary Requirements
    - Start with a clear, direct definition of the topic
    - Focus on what the condition/topic IS, not what the article will cover
    - Avoid phrases like "This article explores..." or "We discuss..."
    - Keep it concise (2-3 sentences maximum)
    - Use active voice and present tense
    - Include key prevalence or impact information if relevant

    Format the response as a JSON object with the following structure:
    {
      "title": "string",
      "slug": "string (URL-friendly version of the title)",
      "summary": "string (2-3 sentences, direct definition only)",
      "category": "${category}",
      "tags": ["string", "string", "string"] (array of relevant tags, e.g. ["mental health", "psychosis", "treatment"]),
      "future_directions": "string (emerging research and future developments)",
      "key_evidence": "string (key research findings and evidence)",
      "references_and_resources": "string (list of key references, resources, and further reading)",
      ${requiredFields.filter(field => !['tags', 'future_directions', 'key_evidence', 'references_and_resources'].includes(field)).map(field => `"${field}": "string"`).join(',\n      ')}
    }
    
    Ensure the content is well-structured with clear headings and subheadings in markdown format.
    The content should be comprehensive but accessible, using clear language and explaining any technical terms.
    Include relevant statistics and research findings where appropriate.
    IMPORTANT: 
    1. The category field must be set to "${category}" exactly as provided.
    2. All fields listed above are required and must be included.
    3. The tags field MUST be an array of strings, not a single string.
    4. All text fields must be strings, not arrays.
    5. The summary must be a direct definition, not a description of what the article covers.
    6. Follow the section order exactly as specified above.
    7. For neurodiversity articles, these fields are REQUIRED:
       - neurodiversity_perspective
       - common_strengths_and_challenges
       - prevalence_and_demographics
       - mechanisms_and_understanding
       - common_misconceptions
       - lived_experience
    8. These base fields are ALWAYS required:
       - future_directions
       - key_evidence
       - references_and_resources`;

  try {
    const completion = await openai.chat.completions.create({
      model: model,
      messages: [
        {
          role: "system",
          content: `You are a mental health content expert. Generate well-researched, accurate, and empathetic content about mental health topics. 
            Always include all required fields in your response exactly as specified in the schema. 
            Ensure the content is comprehensive and evidence-based.
            For summaries, always start with a direct definition of the topic, avoiding phrases like "This article explores..." or "We discuss...".
            Focus on what the condition/topic IS, not what the article will cover.
            For ${category} articles, make sure to include all required fields: ${requiredFields.join(', ')}.
            IMPORTANT: 
            1. The tags field must be an array of strings, not a single string. Example: ["mental health", "psychosis", "treatment"]
            2. Always include future_directions, key_evidence, and references_and_resources fields
            3. For neurodiversity articles, include all neurodiversity-specific fields
            4. For mental health articles, always include references_and_resources with key references and further reading`
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" }
    });

    const content = completion.choices[0].message.content;
    console.log(chalk.yellow('\nDebug: Raw GPT response:'));
    console.log(chalk.yellow(content.substring(0, 200) + '...')); // Log first 200 chars

    let article = JSON.parse(content);

    // Process the article to handle any array responses
    article = processArticleResponse(article);

    // Ensure tags is an array
    if (article.tags && typeof article.tags === 'string') {
      article.tags = article.tags.split(',').map(tag => tag.trim());
    } else if (!Array.isArray(article.tags)) {
      article.tags = [];
    }

    // Add slug if not provided
    if (!article.slug) {
      article.slug = generateSlug(article.title);
    }

    // Log the article before validation
    console.log(chalk.yellow('\nDebug: Article before validation:'));
    console.log(chalk.yellow(JSON.stringify({
      key_evidence: article.key_evidence?.substring(0, 50) + '...',
      practical_takeaways: article.practical_takeaways?.substring(0, 50) + '...'
    }, null, 2)));

    // Validate the generated article
    const validation = validateArticle(article);
    if (!validation.isValid) {
      console.error(chalk.red('\nValidation errors:'));
      validation.errors.forEach(error => {
        console.error(chalk.red(`- ${error.path.join('.')}: ${error.message}`));
        // Log the actual value that caused the error
        console.error(chalk.red(`  Actual value:`, JSON.stringify(error.received)));
      });
      throw new Error('Generated article failed validation');
    }

    // Nest content fields under content_blocks
    const dbReadyArticle = nestContentBlocks(article);

    return dbReadyArticle;
  } catch (error) {
    if (error instanceof SyntaxError) {
      console.error(chalk.red('\nError parsing GPT response as JSON:'), error);
      console.error(chalk.red('Raw response:'), content);
    } else {
      console.error(chalk.red('\nError generating article:'), error);
    }
    throw error;
  }
}

// Main function to handle file operations and article generation
async function main() {
  try {
    // Validate category
    const validCategories = [
      'mental_health', 'neuroscience', 'psychology', 'brain_health',
      'neurodiversity', 'interventions', 'lifestyle_factors', 'lab_testing', 'risk_factors'
    ];
    
    if (!validCategories.includes(options.category)) {
      throw new Error(`Invalid category. Must be one of: ${validCategories.join(', ')}`);
    }

    const filePath = path.join(__dirname, '..', options.output);

    // Only clear the file if --no-clear is not set
    if (options.clear !== false) {
      await fs.writeFile(filePath, '[]');
      console.log(chalk.blue('\nCleared existing articles from output file.'));
    }

    // Generate the article
    const article = await generateArticle(options.topic, options.category, model);
    
    // Set status to published by default
    article.status = 'published';
    
    // Read existing articles
    let articles = [];
    try {
      const existingContent = await fs.readFile(filePath, 'utf-8');
      articles = JSON.parse(existingContent);
      if (!Array.isArray(articles)) {
        articles = [];
      }
    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }

    // Add new article
    articles.push(article);

    // Save updated articles
    await fs.writeFile(filePath, JSON.stringify(articles, null, 2));

    console.log(chalk.green('\nArticle generated successfully!'));
    console.log(chalk.green(`Saved to: ${options.output}`));
    console.log(chalk.green(`Total articles: ${articles.length}`));

  } catch (error) {
    console.error(chalk.red('\nError:'), error.message);
    process.exit(1);
  }
}

// Run the script
main(); 