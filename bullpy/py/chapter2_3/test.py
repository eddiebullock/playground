guest_list = ['your mum', 'greg davies', 'claude', 'jesus']
print(guest_list)

guest_list = ['your mum', 'greg davies', 'claude', 'jesus']
guest_list.pop(0)
print(guest_list)
guest_list.insert(0, 'your dad')
print(guest_list)

guest_list = ['your mum', 'greg davies', 'claude', 'jesus']
guest_list.pop(0)
guest_list.insert(0, 'your dad')
guest_list.insert(2, 'god')
guest_list.append('lucifer')
print("lads the invite list has changed like, more man coming in, plees c the new list -> ", guest_list)

print("peak news team i can only invite the big 2 ")
guest_list.pop(1)
print(guest_list)
print("sorry ", guest_list.pop(1) + " you didnt make the cut.")
guest_list.pop(2)
print("sorry " + guest_list.pop(2) + " you also didnt make the cut")
print("the big two -> ", guest_list)
print("oi ", guest_list[1] + " youre still invited fam")
print("oi ", guest_list[0] + " youre still invited fam")

print(guest_list)
no_guests = len(guest_list)
print("No. guests: ", no_guests)
del guest_list[0]
del guest_list[0]
print(guest_list)


