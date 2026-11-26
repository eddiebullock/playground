import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';
import { redirect } from 'next/navigation';
import type { CookieOptions } from '@supabase/ssr';

export async function getCurrentUser() {
    const cookieStore = cookies();
    const supabase = createServerClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        {
            cookies: {
                get(name: string) {
                    const cookie = cookieStore.get(name);
                    return cookie?.value;
                },
                set(name: string, value: string, options: CookieOptions) {
                    // Server components can't set cookies directly
                    // The middleware will handle this
                },
                remove(name: string, options: CookieOptions) {
                    // Server components can't remove cookies directly
                    // The middleware will handle this
                },
            },
        }
    );

    try {
        const { data: { user }, error } = await supabase.auth.getUser();
        
        if (error || !user) {
            console.log('No user found, redirecting to login');
            redirect('/login');
        }

        return user;
    } catch (error) {
        console.error('Error in getCurrentUser:', error);
        redirect('/login');
    }
}

interface CookieOptions {
    name: string;
    value: string;
    path?: string;
    domain?: string;
    maxAge?: number;
    httpOnly?: boolean;
    secure?: boolean;
    sameSite?: 'strict' | 'lax' | 'none';
} 