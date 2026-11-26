import openai

def generate_communication_report(text):
    prompt = f"""
    Generate a detailed autism assessment report section for the following communication aspects based on the clinician's notes. 
    **DO NOT LEAVE SECTIONS BLANK OR USE PLACEHOLDERS LIKE '[Description]'.** 

    Sections to generate:
    - Gesture
    - Social Imaginative Play
    - Conversational Interchange
    - Repetitive or Unusual Speech

    Each section should be formatted as follows:
    
    **Gesture:**
    Provide a detailed paragraph describing the child’s use of gestures in communication, including whether they point, wave, or use facial expressions. If no information is available, explain why.

    **Social Imaginative Play:**
    Provide a full paragraph on the child's ability to engage in pretend play and social play with peers. If no data is available, mention it explicitly and suggest further assessment.

    **Conversational Interchange:**
    Describe the child's ability to initiate and maintain conversations, including turn-taking and topic coherence. If the clinician's notes lack relevant information, suggest ways to assess this.

    **Repetitive or Unusual Speech:**
    Provide an explanation of any repetitive speech patterns, echolalia, or unusual speech characteristics. If no repetitive speech is reported, explicitly state that and provide general guidance.

    Clinician's Notes:
    ---
    {text}
    ---

    **Ensure each section has a complete and detailed response. If any section is missing details, state this explicitly instead of leaving it blank.**
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1500,  # Increased max tokens for completeness
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']
