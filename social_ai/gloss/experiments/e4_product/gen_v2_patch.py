# Generation prompt v2 (from E4 baseline failure categories, Gricean in form, not item-specific).
# Apply only after the current-engine E4 run finishes; test on the held-out (even) half.
p='/Users/eb2007/playground/social_ai/gloss/app/src/engine/reflect.ts'; s=open(p).read()
old='''If their reaction seems bigger than the situation explains, one reading can be that it is about something on their side, not about me.'''
new='''Look at what they did not say as well as what they said. If the words do not answer what was asked, or give far more or far less detail than needed, one reading must say what that suggests (for example: they would rather not talk about it, or they did not like it but do not want to say so). If they used a figure of speech, say plainly what it means about the person or thing it describes.
If their reaction seems bigger than the situation explains, one reading can be that it is about something on their side, not about me.'''
assert s.count(old)==1; s=s.replace(old,new); open(p,'w').write(s); print('patched')
