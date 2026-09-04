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

# The correct approach: use get_follows to get profiles, then unfollow each one
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
        res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'cursor': cursor, 'limit': 100})
        
        if not res.follows:
            print("No follows found in this batch.")
            if not res.cursor:
                print("No more follows to process.")
                break
            else:
                cursor = res.cursor
                continue
        
        print(f"Found {len(res.follows)} follows in this batch")
        
        # Process each follow in this batch
        for i, follow in enumerate(res.follows):
            if unfollowed >= MAX_UNFOLLOWS_PER_SESSION:
                break
                
            processed += 1
            print(f"Unfollowing {follow.handle} ({processed})")
            
            try:
                # Get the DID for this handle
                did = follow.did
                
                # Use the correct unfollow method - we need to find the follow record first
                # Let's try using the graph.unfollow method if it exists
                try:
                    # Try the direct unfollow approach
                    client.app.bsky.graph.unfollow({'subject': did})
                    unfollowed += 1
                    print(f"  ✓ Successfully unfollowed {follow.handle}")
                    
                except AttributeError:
                    # If unfollow method doesn't exist, we need to find and delete the record
                    # This is more complex - let's try a different approach
                    print(f"  ⚠️  Direct unfollow not available, trying alternative method...")
                    
                    # Try to get the follow records for this specific account
                    # We'll need to search through our follow records
                    follow_records = client.com.atproto.repo.list_records({
                        'repo': client.me.did,
                        'collection': 'app.bsky.graph.follow',
                        'limit': 100
                    })
                    
                    # Find the record for this specific DID
                    target_record = None
                    for record in follow_records.records:
                        if hasattr(record, 'value') and hasattr(record.value, 'subject'):
                            if record.value.subject == did:
                                target_record = record
                                break
                    
                    if target_record:
                        # Delete the follow record
                        client.com.atproto.repo.delete_record({
                            'repo': client.me.did,
                            'collection': 'app.bsky.graph.follow',
                            'rkey': target_record.uri.split('/')[-1]
                        })
                        unfollowed += 1
                        print(f"  ✓ Successfully unfollowed {follow.handle}")
                    else:
                        print(f"  ⚠️  Could not find follow record for {follow.handle}")
                        continue
                
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
