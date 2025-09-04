const { BskyAgent } = require('@atproto/api');

const agent = new BskyAgent({ service: 'https://bsky.social' });

const username = 'eddiebullock.bsky.social';
const password = 'akul-5hfw-qksm-cucn'; // App password

(async () => {
  try {
    await agent.login({ identifier: username, password });
    const did = agent.session.did;

    const res = await agent.api.app.bsky.graph.getFollows({
      actor: did,
      limit: 10, // just check 10 for now
    });

    const follows = res.data.follows;

    if (!follows || follows.length === 0) {
      console.log("⚠️ API thinks you're following nobody.");
    } else {
      console.log("✅ You're following:");
      follows.forEach((user, i) => {
        console.log(`${i + 1}. ${user.displayName || user.handle}`);
      });
    }
  } catch (err) {
    console.error('Error:', err);
  }
})();

