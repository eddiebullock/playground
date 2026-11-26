import openai

def generate_rrb_report(text):
    prompt = f"""
    Generate a detailed autism assessment report section for the following topics based on the clinician's notes. 
    **DO NOT USE PLACEHOLDERS LIKE '[Description]'.** Ensure each section has a detailed response.

    Sections to generate:
    - Preoccupation
    - Routines
    - Repetitive Movements
    - Sensory Sensitivities
    - Development at and Before 36 Months

    Each section should be formatted as follows:
    
    **Preoccupation:**
    Provide a complete paragraph describing the child's preoccupation behaviors.

    **Routines:**
    Provide a complete paragraph detailing how the child responds to routines, any challenges, and required support.

    **Repetitive Movements:**
    Provide a detailed explanation of the child’s repetitive movements and behaviors.

    **Sensory Sensitivities:**
    Provide a full section on any sensory sensitivities observed.

    **Development at and Before 36 Months:**
    Provide detailed information on early developmental concerns, including speech, motor skills, and behavior.

    Clinician's Notes:
    ---
    {text}
    ---

    **Ensure every section has a complete response and is not left empty.**
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1000,  # Increase max tokens
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']
