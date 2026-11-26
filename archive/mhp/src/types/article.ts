export type ArticleCategory =
  | 'mental_health'
  | 'neuroscience'
  | 'psychology'
  | 'brain_health'
  | 'neurodiversity'
  | 'interventions'
  | 'lifestyle_factors'
  | 'lab_testing';

export interface BaseArticle {
  id: string;
  title: string;
  summary: string;
  category: ArticleCategory;
  content_blocks: {
    overview?: string;
    mechanisms?: string;
    practical_takeaways?: string;
    [key: string]: string | undefined;
  };
  tags: string[];
  status: 'published' | 'draft' | 'archived';
  created_at: string;
  updated_at: string;
}

export interface MentalHealthArticle extends BaseArticle {
  category: 'mental_health';
  content_blocks: BaseArticle['content_blocks'] & {
    prevalence?: string;
    causes_and_mechanisms?: string;
    symptoms_and_impact?: string;
    evidence_summary?: string;
    practical_takeaways?: string;
    common_myths?: string;
  };
}

export interface NeuroscienceArticle extends BaseArticle {
  category: 'neuroscience';
  content_blocks: BaseArticle['content_blocks'] & {
    definition?: string;
    mechanisms?: string;
    relevance?: string;
    key_studies?: string;
    common_misconceptions?: string;
    practical_implications?: string;
  };
}

export interface InterventionArticle extends BaseArticle {
  category: 'interventions';
  content_blocks: BaseArticle['content_blocks'] & {
    how_it_works?: string;
    evidence_base?: string;
    effectiveness?: string;
    practical_applications?: string;
    common_myths?: string;
    risks_and_limitations?: string;
  };
}

export interface LifestyleArticle extends BaseArticle {
  category: 'lifestyle_factors';
  content_blocks: BaseArticle['content_blocks'] & {
    how_it_works?: string;
    evidence_base?: string;
    effectiveness?: string;
    practical_applications?: string;
    common_myths?: string;
    risks_and_limitations?: string;
  };
}

export interface LabTestingArticle extends BaseArticle {
  category: 'lab_testing';
  content_blocks: BaseArticle['content_blocks'] & {
    how_it_works?: string;
    applications?: string;
    strengths_and_limitations?: string;
    risks_and_limitations?: string;
  };
}

export type Article = MentalHealthArticle | NeuroscienceArticle | InterventionArticle | LifestyleArticle | LabTestingArticle; 