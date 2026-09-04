from atproto import Client
import getpass

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

# Get a few follows to examine the structure
print("Fetching follows to examine structure...")
res = client.app.bsky.graph.get_follows({'actor': client.me.did, 'limit': 5})

if res.follows:
    follow = res.follows[0]
    print(f"\nFollow object type: {type(follow)}")
    print(f"Follow object attributes: {dir(follow)}")
    print(f"\nFollow object: {follow}")
    
    # Try to access different possible attributes
    for attr in ['uri', 'record', 'created_at', 'indexed_at']:
        if hasattr(follow, attr):
            print(f"follow.{attr}: {getattr(follow, attr)}")
        else:
            print(f"follow.{attr}: NOT FOUND")
else:
    print("No follows found")
