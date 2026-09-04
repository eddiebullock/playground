from atproto import Client
import getpass

# Securely enter login info
handle = "eddiebullock.bsky.social"
password = getpass.getpass("Enter your Bluesky password: ")

# Login
client = Client()
client.login(handle, password)

# Fetch follows
cursor = None
follows = []

while True:
    res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'cursor': cursor})
    follows.extend(res.follows)
    cursor = res.cursor
    if not cursor:
        break

print(f"You follow {len(follows)} accounts")

for f in follows[:50]:  # show first 50
    print(f.handle)

