sentence = "well, its about 200 meters. well it isnt about 200 meters."
words = sentence.split()    
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1
print(word_count)

a = set(words)
print(a)