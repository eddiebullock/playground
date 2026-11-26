import { createClient } from 'redis';

const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const client = createClient({ url: REDIS_URL });
client.connect();

const CACHE_TTL = 60 * 60 * 24; // 24 hours

export const cacheManager = {
  async get(key: string) {
    try {
      const value = await client.get(key);
      return value ? JSON.parse(value) : null;
    } catch {
      return null;
    }
  },
  async set(key: string, value: any) {
    try {
      await client.set(key, JSON.stringify(value), { EX: CACHE_TTL });
    } catch {
      // ignore
    }
  },
}; 