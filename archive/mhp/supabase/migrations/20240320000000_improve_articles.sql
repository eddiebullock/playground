-- Add index for slug (for fast lookups)
create index if not exists idx_articles_slug on articles (slug);

-- Add index for category_id (for filtering by category)
create index if not exists idx_articles_category_id on articles (category_id);

-- Add index for tags (if frequently used for filtering)
create index if not exists idx_articles_tags on articles using gin (tags);

-- Add a constraint for status (only if it doesn't exist)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'status_check' 
        AND conrelid = 'articles'::regclass
    ) THEN
        alter table articles
          add constraint status_check
          check (status in ('published', 'draft', 'archived'));
    END IF;
END $$;

-- Set default for created_at and updated_at
alter table articles
  alter column created_at set default now(),
  alter column updated_at set default now();

-- Add trigger to automatically update updated_at
create or replace function update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists update_articles_updated_at on articles;
create trigger update_articles_updated_at
  before update on articles
  for each row
  execute function update_updated_at_column(); 