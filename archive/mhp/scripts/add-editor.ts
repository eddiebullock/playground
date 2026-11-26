import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function addEditor(email: string) {
  try {
    console.log(`Looking up user with email: ${email}`);
    
    // First, find the user by email
    const { data: users, error: userError } = await supabase.auth.admin.listUsers();
    
    if (userError) {
      console.error('Error finding users:', userError);
      return;
    }
    
    const user = users.users.find(u => u.email === email);
    
    if (!user) {
      console.error('User not found');
      return;
    }
    
    console.log(`Found user: ${user.id} (${user.email})`);
    
    // Check if user is already an editor
    const { data: existingEditor, error: checkError } = await supabase
      .from('article_editors')
      .select('id')
      .eq('user_id', user.id)
      .single();
    
    if (checkError && checkError.code !== 'PGRST116') {
      console.error('Error checking existing editor status:', checkError);
      return;
    }
    
    if (existingEditor) {
      console.log('User is already an editor');
      return;
    }
    
    // Add user as editor
    const { data: editor, error: insertError } = await supabase
      .from('article_editors')
      .insert({
        user_id: user.id,
        created_by: user.id // Self-created for now
      })
      .select()
      .single();
    
    if (insertError) {
      console.error('Error adding editor:', insertError);
      return;
    }
    
    console.log('Successfully added user as editor:', editor);
    
  } catch (error) {
    console.error('Unexpected error:', error);
  }
}

// Get email from command line argument
const email = process.argv[2];

if (!email) {
  console.error('Please provide an email address as an argument');
  console.error('Usage: npx tsx scripts/add-editor.ts user@example.com');
  process.exit(1);
}

addEditor(email); 