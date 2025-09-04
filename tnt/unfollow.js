const { BskyAgent } = require('@atproto/api');

const agent = new BskyAgent({ service: 'https://bsky.social' });

// Replace with your handle and app password (NOT your login password)
const username = 'eddiebullock.bsky.social';
const password = 'akul-5hfw-qksm-cucn'; // <- Your Bluesky App Password

(async () => {
  try {
    await agent.login({ identifier: username, password });
    const did = agent.session.did;

    let cursor = null;
    let totalUnfollowed = 0;

    while (true) {
      const res = await agent.api.app.bsky.graph.getFollows({
        actor: did,
        limit: 100,
        cursor,
      });

      const follows = res.data.follows;
      if (!follows || follows.length === 0) break;

      for (const user of follows) {
        const rkey = user.uri.split('/').pop();

        try {
          await agent.api.app.bsky.graph.unfollow({ repo: did, rkey });
          console.log(`🚫 Unfollowed ${user.displayName || user.handle}`);
          totalUnfollowed++;
        } catch (err) {
          console.error(`❌ Failed to unfollow ${user.handle}:`, err.message);
        }
      }

      cursor = res.data.cursor;
      if (!cursor) break;
    }

    console.log(`🎉 Finished. Total unfollowed: ${totalUnfollowed}`);
  } catch (err) {
    console.error('Login failed or other error:', err);
  }
})();

