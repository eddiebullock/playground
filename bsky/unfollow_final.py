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
MIN_DELAY = 0.3
MAX_DELAY = 1.5
BATCH_SIZE = 50
BATCH_DELAY = 3

print(f"\nBot Protection Settings:")
print(f"  - Continuous mode: Will run until manually stopped (Ctrl+C)")
print(f"  - Delay between actions: {MIN_DELAY}-{MAX_DELAY} seconds")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Delay between batches: {BATCH_DELAY} seconds")

confirm = input(f"\nDo you want to start continuous unfollowing? (yes/no): ")
if confirm.lower() != 'yes':
    print("Cancelled.")
    exit(0)

print("\n⚠️  CONTINUOUS MODE ACTIVE")
print("The script will run until you stop it with Ctrl+C")
print("Press Ctrl+C when you want to stop unfollowing")

# Continuous unfollowing loop
session_count = 0
total_unfollowed = 0
total_failed = 0

try:
    while True:
        session_count += 1
        print(f"\n🔄 Starting session {session_count}")
        
        # Fetch all follow records for this session
        print("Fetching all follow records...")
        cursor = None
        all_follow_records = []
        batch_num = 0

        while True:
            batch_num += 1
            print(f"Fetching follow records batch {batch_num}...")
            
            try:
                records = client.com.atproto.repo.list_records({
                    'repo': client.me.did,
                    'collection': 'app.bsky.graph.follow',
                    'cursor': cursor,
                    'limit': 100
                })
                
                all_follow_records.extend(records.records)
                print(f"Found {len(records.records)} follow records (total: {len(all_follow_records)})")
                
                cursor = records.cursor
                if not cursor:
                    break
                    
            except Exception as e:
                print(f"Error fetching follow records: {e}")
                break

        print(f"\nTotal follow records found: {len(all_follow_records)}")

        if not all_follow_records:
            print("No follow records found. Waiting 30 seconds before checking again...")
            time.sleep(30)
            continue

        # Unfollow each record
        session_unfollowed = 0
        session_failed = 0

        for i, record in enumerate(all_follow_records):
            try:
                # Get the handle from the record
                if hasattr(record, 'value') and hasattr(record.value, 'subject'):
                    subject_did = record.value.subject
                    
                    # Get the profile to find the handle
                    try:
                        profile = client.app.bsky.actor.get_profile({'actor': subject_did})
                        handle_name = profile.handle
                    except:
                        handle_name = subject_did
                    
                    print(f"Unfollowing {handle_name} ({i+1}/{len(all_follow_records)})")
                    
                    # Delete the follow record
                    client.com.atproto.repo.delete_record({
                        'repo': client.me.did,
                        'collection': 'app.bsky.graph.follow',
                        'rkey': record.uri.split('/')[-1]
                    })
                    
                    session_unfollowed += 1
                    total_unfollowed += 1
                    print(f"  ✓ Successfully unfollowed {handle_name}")
                    
                    # Random delay between actions
                    delay = random.uniform(MIN_DELAY, MAX_DELAY)
                    print(f"  ⏳ Waiting {delay:.1f} seconds...")
                    time.sleep(delay)
                    
                    # Batch delay every BATCH_SIZE unfollows
                    if session_unfollowed % BATCH_SIZE == 0:
                        print(f"\n⏳ Batch complete. Waiting {BATCH_DELAY} seconds...")
                        time.sleep(BATCH_DELAY)
                    
                else:
                    print(f"  ⚠️  Skipping record {i+1} - invalid structure")
                    continue
                    
            except Exception as e:
                session_failed += 1
                total_failed += 1
                print(f"  ✗ Failed to unfollow record {i+1}: {e}")
                
                # Still wait even on failure to maintain timing
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)

        # Session complete
        print(f"\n📊 Session {session_count} complete!")
        print(f"Session unfollowed: {session_unfollowed}")
        print(f"Session failed: {session_failed}")
        print(f"Total unfollowed: {total_unfollowed}")
        print(f"Total failed: {total_failed}")
        
        # Wait before next session
        print(f"\n⏳ Waiting 10 seconds before next session...")
        time.sleep(10)

except KeyboardInterrupt:
    print(f"\n\n🛑 STOPPED BY USER")
    print(f"📊 Final Statistics:")
    print(f"Total sessions: {session_count}")
    print(f"Total unfollowed: {total_unfollowed}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {(total_unfollowed/(total_unfollowed+total_failed)*100):.1f}%" if (total_unfollowed+total_failed) > 0 else "N/A")
