def build_profile(first, last, **user_info):
    """build a dict containing everything we know about user"""
    profile = {}
    profile['first_name'] = first
    profile['last_name'] = last
    for key, value in user_info.items():
        profile[key] = value
    return profile

user_profile = build_profile('albert', 'einstein', location='princeton', field='physics')
user_profile_1 = build_profile('the', 'bull', casa='the pen', food='bants')
print(user_profile) 
print(user_profile_1)
