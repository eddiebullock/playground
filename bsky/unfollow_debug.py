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

# Bot detection protection settings
MAX_UNFOLLOWS_PER_SESSION = 1000
MIN_DELAY = 0.3
MAX_DELAY = 1.5
BATCH_SIZE = 50
BATCH_DELAY = 3

print(f"\nBot Protection Settings:")
print(f"  - Max unfollows per session: {MAX_UNFOLLOWS_PER_SESSION}")
print(f"  - Delay between actions: {MIN_DELAY}-{MAX_DELAY} seconds")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Delay between batches: {BATCH_DELAY} seconds")

confirm = input(f"\nDo you want to unfollow up to {MAX_UNFOLLOWS_PER_SESSION} accounts? (yes/no): ")
if confirm.lower() != 'yes':
    print("Cancelled.")
    exit(0)

# Now let's properly fetch and unfollow
unfollowed = 0
failed = 0
processed = 0

cursor = None
batch_num = 0

print("\nStarting unfollow process...")

while unfollowed < MAX_UNFOLLOWS_PER_SESSION:
    batch_num += 1
    print(f"\n--- Fetching batch {batch_num} ---")
    
    try:
        # Try different limit values to see what works
        res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'cursor': cursor, 'limit': 100})
        
        print(f"API Response: {res}")
        print(f"Number of follows in response: {len(res.follows)}")
        print(f"Cursor: {res.cursor}")
        
        if not res.follows:
            print("No follows found in this batch.")
            if not res.cursor:
                print("No more follows to process.")
                break
            else:
                print("Continuing with next batch...")
                cursor = res.cursor
                continue
        
        # Process each follow in this batch
        for i, follow in enumerate(res.follows):
            if unfollowed >= MAX_UNFOLLOWS_PER_SESSION:
                break
                
            processed += 1
            print(f"Unfollowing {follow.handle} ({processed})")
            
            try:
                # Debug: let's see what the follow object contains
                print(f"  Follow object: {follow}")
                print(f"  Follow type: {type(follow)}")
                print(f"  Follow attributes: {[attr for attr in dir(follow) if not attr.startswith('_')]}")
                
                # The follow object should have the URI - let's find it
                if hasattr(follow, 'uri'):
                    uri = follow.uri
                elif hasattr(follow, 'record') and hasattr(follow.record, 'uri'):
                    uri = follow.record.uri
                else:
                    print(f"  ⚠️  Could not find URI for {follow.handle}")
                    continue
                
                print(f"  Found URI: {uri}")
                
                # Extract the record key from the URI
                record_key = uri.split('/')[-1]
                print(f"  Record key: {record_key}")
                
                # Delete the follow record
                client.com.atproto.repo.delete_record({
                    'repo': client.me.did,
                    'collection': 'app.bsky.graph.follow',
                    'rkey': record_key
                })
                
                unfollowed += 1
                print(f"  ✓ Successfully unfollowed {follow.handle}")
                
                # Random delay between actions
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                print(f"  ⏳ Waiting {delay:.1f} seconds...")
                time.sleep(delay)
                
                # Batch delay every BATCH_SIZE unfollows
                if unfollowed % BATCH_SIZE == 0 and unfollowed < MAX_UNFOLLOWS_PER_SESSION:
                    print(f"\n⏳ Batch complete. Waiting {BATCH_DELAY} seconds...")
                    time.sleep(BATCH_DELAY)
                
            except Exception as e:
                failed += 1
                print(f"  ✗ Failed to unfollow {follow.handle}: {e}")
                
                # Still wait even on failure to maintain timing
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
        
        cursor = res.cursor
        if not cursor:
            print("No more follows to process.")
            break
            
    except Exception as e:
        print(f"Error fetching follows: {e}")
        break

print(f"\nUnfollow complete!")
print(f"Successfully unfollowed: {unfollowed}")
print(f"Failed: {failed}")
print(f"Total processed: {processed}")

if unfollowed > 0:
    print(f"\nYou can run this script again to continue unfollowing more accounts.")
