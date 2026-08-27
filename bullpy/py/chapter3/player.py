players = ['alice', 'joff', 'max', 'alex']
print(players[0:3])
print(players[1:4])
print(players[:4])
print(players[2:])

print("here are the first three player: ")
for player in players[:3]:
    print(player.title())