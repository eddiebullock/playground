from atproto import Client
import getpass
import time
import random

# Securely enter login info
handle = "eddiebullock.bsky.social"
try:
    password = getpass.getpass("password: ")
except (EOFError, KeyboardInterrupt):
    print("\nPassword input cancelled. Please run the script in a proper terminal.")
    exit(1)

# Login
print("Attempting to login...")
client = Client()
try:
    client.login(handle, password)
    print("Login successful!")
except Exception as e:
    print(f"Login failed: {e}")
    exit(1)

# Read handles to unfollow from file
try:
    with open('follows_list.txt', 'r') as f:
        handles_to_unfollow = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(handles_to_unfollow)} handles from follows_list.txt")
except FileNotFoundError:
    print("follows_list.txt not found. Run bsky1.py first to generate the list.")
    exit(1)

# Show first 10 handles
print(f"\nFirst 10 handles to unfollow:")
for handle in handles_to_unfollow[:10]:
    print(f"  {handle}")

# Bot detection protection settings
MAX_UNFOLLOWS_PER_SESSION = 1000  # Limit per session
MIN_DELAY = 0.3  # Minimum seconds between actions (300ms)
MAX_DELAY = 1.5  # Maximum seconds between actions (1.5s)
BATCH_SIZE = 50  # Smaller batches for better performance
BATCH_DELAY = 3  # Shorter delay between batches

print(f"\nBot Protection Settings:")
print(f"  - Max unfollows per session: {MAX_UNFOLLOWS_PER_SESSION}")
print(f"  - Delay between actions: {MIN_DELAY}-{MAX_DELAY} seconds")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Delay between batches: {BATCH_DELAY} seconds")

# Limit the number of unfollows
if len(handles_to_unfollow) > MAX_UNFOLLOWS_PER_SESSION:
    handles_to_unfollow = handles_to_unfollow[:MAX_UNFOLLOWS_PER_SESSION]
    print(f"\nLimited to first {MAX_UNFOLLOWS_PER_SESSION} accounts for safety")

# Confirm before proceeding
confirm = input(f"\nDo you want to unfollow these {len(handles_to_unfollow)} accounts? (yes/no): ")
if confirm.lower() != 'yes':
    print("Cancelled.")
    exit(0)

# Get all current follows once and create a lookup map
print("Fetching current follows (this may take a moment)...")
cursor = None
follows_map = {}
batch_count = 0

while True:
    batch_count += 1
    print(f"Fetching batch {batch_count}...")
    res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'cursor': cursor, 'limit': 100})
    
    for follow in res.follows:
        follows_map[follow.did] = follow
    
    cursor = res.cursor
    if not cursor:
        break

print(f"Found {len(follows_map)} current follows")

# Unfollow accounts
unfollowed = 0
failed = 0
not_following = 0

for i, handle_to_unfollow in enumerate(handles_to_unfollow):
    try:
        print(f"Unfollowing {handle_to_unfollow} ({i+1}/{len(handles_to_unfollow)})")
        
        # Get the DID for this handle
        profile = client.app.bsky.actor.get_profile({'actor': handle_to_unfollow})
        did = profile.did
        
        # Check if we're following this account
        if did not in follows_map:
            not_following += 1
            print(f"  ⚠️  Not following {handle_to_unfollow} (skipping)")
            continue
        
        follow_record = follows_map[did]
        
        # Delete the follow record
        client.com.atproto.repo.delete_record({
            'repo': client.me.did,
            'collection': 'app.bsky.graph.follow',
            'rkey': follow_record.uri.split('/')[-1]
        })
        
        unfollowed += 1
        print(f"  ✓ Successfully unfollowed {handle_to_unfollow}")
        
        # Random delay between actions
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        print(f"  ⏳ Waiting {delay:.1f} seconds...")
        time.sleep(delay)
        
        # Batch delay every BATCH_SIZE unfollows
        if (i + 1) % BATCH_SIZE == 0 and i + 1 < len(handles_to_unfollow):
            print(f"\n⏳ Batch complete. Waiting {BATCH_DELAY} seconds...")
            time.sleep(BATCH_DELAY)
        
    except Exception as e:
        failed += 1
        print(f"  ✗ Failed to unfollow {handle_to_unfollow}: {e}")
        
        # Still wait even on failure to maintain timing
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)

print(f"\nUnfollow complete!")
print(f"Successfully unfollowed: {unfollowed}")
print(f"Failed: {failed}")
print(f"Not following (skipped): {not_following}")

# Save remaining handles for next session
if unfollowed > 0:
    remaining_handles = handles_to_unfollow[unfollowed + failed + not_following:]
    if remaining_handles:
        with open('remaining_follows.txt', 'w') as f:
            for handle in remaining_handles:
                f.write(f"{handle}\n")
        print(f"Saved {len(remaining_handles)} remaining handles to 'remaining_follows.txt'")
        print("You can run the script again later to continue unfollowing.")
