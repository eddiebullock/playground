import openai

def generate_rsi_report(text):
    prompt = f"""
    Generate the following sections of an autism assessment report based on the provided clinician's notes. 
    **Ensure each section title is formatted in bold and has a clear line break.** 
    Do not use numbered lists.

    Sections:
    - Nonverbal Behaviors to Regulate Emotion
    - Developing Peer Relationships
    - Shared Enjoyment
    - Socioemotional Reciprocity

    **Format Example:**
    Nonverbal Behaviors to Regulate Emotion:
    [Description]

    Developing Peer Relationships:
    [Description]

    Shared Enjoyment:
    [Description]

    Socioemotional Reciprocity:
    [Description]

    Clinician's Notes:
    ---
    {text}
    ---

    Ensure each section follows the format where the title is bolded and followed by a paragraph explaining the findings. Do not include numbering (1., 2., etc.).
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=700,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']

