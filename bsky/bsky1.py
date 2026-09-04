from atproto import Client
import getpass

# Securely enter login info
handle = "eddiebullock.bsky.social"
password = getpass.getpass("password: ")

# Login
print("Attempting to login...")
client = Client()
try:
    client.login(handle, password)
    print("Login successful!")
except Exception as e:
    print(f"Login failed: {e}")
    exit(1)

# Fetch follows
print("Fetching follows...")
cursor = None
follows = []
max_follows = 10000  # Get up to 10k follows

batch_num = 0
while len(follows) < max_follows:
    batch_num += 1
    print(f"Fetching batch {batch_num} with cursor: {cursor}")
    try:
        res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'cursor': cursor, 'limit': 100})
        follows.extend(res.follows)
        print(f"Got {len(res.follows)} follows in this batch (total: {len(follows)})")
        cursor = res.cursor
        if not cursor:
            break
    except Exception as e:
        print(f"Error fetching follows: {e}")
        break

print(f"\nStopped at {len(follows)} follows (limit: {max_follows})")

# Save follows to file
with open('follows_list.txt', 'w') as f:
    for follow in follows:
        f.write(f"{follow.handle}\n")

print(f"Saved {len(follows)} follows to 'follows_list.txt'")

# Show first 20 follows
print(f"\nFirst 20 follows:")
for f in follows[:20]:
    print(f"  {f.handle}")

