export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type ArticleCategory =
  | 'mental_health'
  | 'neuroscience'
  | 'psychology'
  | 'brain_health'
  | 'neurodiversity'
  | 'interventions'
  | 'lifestyle_factors'
  | 'lab_testing'
  | 'risk_factors';

export interface Database {
  public: {
    Tables: {
      articles: {
        Row: {
          id: string
          created_at: string
          updated_at: string
          title: string
          slug: string
          summary: string
          category: ArticleCategory
          status: 'published' | 'draft' | 'archived'
          tags: string[]
          content_blocks: {
            overview?: string
            definition?: string
            mechanisms?: string
            prevalence?: string
            causes_and_mechanisms?: string
            symptoms_and_impact?: string
            evidence_summary?: string
            common_myths?: string
            practical_takeaways?: string
            future_directions?: string
            references_and_resources?: string
            relevance?: string
            key_studies?: string
            common_misconceptions?: string
            practical_implications?: string
            effectiveness?: string
            risks_and_limitations?: string
            how_it_works?: string
            applications?: string
            strengths_and_limitations?: string
            evidence_base?: string
            practical_applications?: string
            neurodiversity_perspective?: string
            common_strengths_and_challenges?: string
            prevalence_and_demographics?: string
            mechanisms_and_understanding?: string
            lived_experience?: string
            core_principles?: string
            key_studies_and_theories?: string
            reliability_score?: string
          }
        }
        Insert: {
          id?: string
          created_at?: string
          updated_at?: string
          title: string
          slug: string
          summary: string
          category: ArticleCategory
          status?: 'published' | 'draft' | 'archived'
          tags?: string[]
          content_blocks?: {
            overview?: string
            definition?: string
            mechanisms?: string
            prevalence?: string
            causes_and_mechanisms?: string
            symptoms_and_impact?: string
            evidence_summary?: string
            common_myths?: string
            practical_takeaways?: string
            future_directions?: string
            references_and_resources?: string
            relevance?: string
            key_studies?: string
            common_misconceptions?: string
            practical_implications?: string
            effectiveness?: string
            risks_and_limitations?: string
            how_it_works?: string
            applications?: string
            strengths_and_limitations?: string
            evidence_base?: string
            practical_applications?: string
            neurodiversity_perspective?: string
            common_strengths_and_challenges?: string
            prevalence_and_demographics?: string
            mechanisms_and_understanding?: string
            lived_experience?: string
            core_principles?: string
            key_studies_and_theories?: string
            reliability_score?: string
          }
        }
        Update: {
          id?: string
          created_at?: string
          updated_at?: string
          title?: string
          slug?: string
          summary?: string
          category?: ArticleCategory
          status?: 'published' | 'draft' | 'archived'
          tags?: string[]
          content_blocks?: {
            overview?: string
            definition?: string
            mechanisms?: string
            prevalence?: string
            causes_and_mechanisms?: string
            symptoms_and_impact?: string
            evidence_summary?: string
            common_myths?: string
            practical_takeaways?: string
            future_directions?: string
            references_and_resources?: string
            relevance?: string
            key_studies?: string
            common_misconceptions?: string
            practical_implications?: string
            effectiveness?: string
            risks_and_limitations?: string
            how_it_works?: string
            applications?: string
            strengths_and_limitations?: string
            evidence_base?: string
            practical_applications?: string
            neurodiversity_perspective?: string
            common_strengths_and_challenges?: string
            prevalence_and_demographics?: string
            mechanisms_and_understanding?: string
            lived_experience?: string
            core_principles?: string
            key_studies_and_theories?: string
            reliability_score?: string
          }
        }
      }
      categories: {
        Row: {
          id: string
          name: string
          description: string | null
        }
        Insert: {
          id?: string
          name: string
          description?: string | null
        }
        Update: {
          id?: string
          name?: string
          description?: string | null
        }
      }
      saved_articles: {
        Row: {
          id: string
          user_id: string
          article_id: string
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          article_id: string
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          article_id?: string
          created_at?: string
        }
      }
      article_edits: {
        Row: {
          id: string
          article_id: string
          editor_id: string
          edited_at: string
          created_at: string
        }
        Insert: {
          id?: string
          article_id: string
          editor_id: string
          edited_at?: string
          created_at?: string
        }
        Update: {
          id?: string
          article_id?: string
          editor_id?: string
          edited_at?: string
          created_at?: string
        }
      }
      article_views: {
        Row: {
          id: string
          article_id: string
          viewer_id: string | null
          viewed_at: string
          created_at: string
        }
        Insert: {
          id?: string
          article_id: string
          viewer_id?: string | null
          viewed_at?: string
          created_at?: string
        }
        Update: {
          id?: string
          article_id?: string
          viewer_id?: string | null
          viewed_at?: string
          created_at?: string
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      get_editor_stats: {
        Args: {
          editor_user_id: string
        }
        Returns: {
          total_edits: number
          unique_articles_edited: number
          total_views: number
          total_saves: number
          editor_score: number
        }[]
      }
      get_editor_rankings: {
        Args: Record<string, never>
        Returns: {
          editor_id: string
          editor_email: string
          total_edits: number
          unique_articles_edited: number
          total_views: number
          total_saves: number
          editor_score: number
          rank_position: number
        }[]
      }
    }
    Enums: {
      [_ in never]: never
    }
  }
} 