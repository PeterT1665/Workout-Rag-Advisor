// Supabase client (accounts + sync). Loaded lazily so guests never download it.
// The URL and publishable key are public by design; row-level security protects each user's rows.
import type { SupabaseClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string | undefined

export const accountsEnabled = !!url && !!key && import.meta.env.MODE !== 'test'

let client: Promise<SupabaseClient> | null = null
export function supabase(): Promise<SupabaseClient> {
  client ??= import('@supabase/supabase-js').then(({ createClient }) =>
    createClient(url!, key!, { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } }))
  return client
}

// Set before the client reads the URL: an invite or password-reset link should land on "set your password".
export const arrivedToSetPassword = typeof location !== 'undefined' && /type=(recovery|invite)/.test(location.hash)
