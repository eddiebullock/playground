"""Author-written E/S narratives. Not EQ-60 or SQ-R item text.

These are original short first-person passages used only to build
contrastive residual-stream directions. They describe a cognitive
emphasis (people's minds vs systems and mechanisms). They are not
questionnaires and they are not trait labels for models.
"""

from __future__ import annotations

# Empathizing-direction narratives: attending to mental states, feelings,
# and interpersonal nuance.
E_NARRATIVES: list[str] = [
    "When two people fall quiet after a joke, I replay their faces and try to tell whether the silence is shared amusement or someone feeling left out.",
    "I notice how a colleague's voice thins when a deadline is mentioned, and I adjust what I ask next so they do not have to perform calm.",
    "In a crowded kitchen I track who has not spoken yet and look for a gap where they could join without being put on the spot.",
    "If a friend says they are fine but their messages get shorter, I treat the shortness as information about how they feel, not as a lack of news.",
    "I spend more time wondering what a stranger intended by a glance than whether the glance followed a rule.",
    "During an argument I listen for the worry underneath the complaint, because the worry usually explains the heat.",
    "I remember birthdays by the feeling someone had last year, not by a calendar algorithm.",
    "When a child hides a drawing, I ask what they hoped I would see rather than how the drawing is constructed.",
    "I can sit with someone who is sad without offering a plan. The sitting is the point.",
    "If a group laughs and one person smiles late, I check whether they understood the reference or were protecting themselves.",
    "I rewrite an email until it sounds like I have considered the other person's day, not just my request.",
    "Watching a film, I stay with what a character is afraid of losing, even when the plot is about machines.",
    "I notice small kindnesses that are easy to miss: a cup moved closer, a pause that lets someone finish.",
    "When someone changes the subject, I ask myself what the last subject cost them.",
    "I am pulled toward faces more than toward diagrams. A face tells me whether to slow down.",
    "If two friends disagree, I try to hold both accounts of how it felt rather than pick the cleaner story.",
    "I remember the exact phrase that made someone flinch, and I do not reuse it.",
    "In meetings I watch who is agreeing with their eyes and who is only agreeing with their mouth.",
    "I would rather understand why a person withdrew than map the workflow that failed.",
    "When praise is public and thin, I look at the person being praised to see if it landed as warmth or as pressure.",
    "I treat tone as data. The same sentence can comfort or cut depending on how it is held.",
    "If someone brings food, I think about what they guessed I needed, not about the recipe's structure.",
    "I notice when a joke is a bid for closeness and when it is a wall.",
    "A delayed reply makes me wonder about their state of mind before I wonder about the mail server.",
    "I am interested in how people recover after they have been misunderstood.",
    "When a stranger holds a door and then looks away, I read the look as shyness more often than as rule-following.",
    "I keep a private catalogue of what makes particular people feel safe enough to be specific.",
    "If a story has a villain, I still ask what the villain thought they were protecting.",
    "I change my walking pace to match the person beside me before I think about the route.",
    "Hearing a raised voice, I scan for fear or humiliation before I scan for who is correct.",
    "I like conversations where we name the feeling in the room and then decide what to do.",
    "When someone over-explains, I hear the wish not to be a burden more than I hear the extra facts.",
    "I remember how a room felt after a hard conversation longer than I remember the agenda.",
    "I try to leave people with a sense that their inner weather was noticed.",
    "If a partner goes quiet over dinner, I set down my own story and ask what arrived for them.",
    "I am curious about the moment a person decides they are not welcome.",
    "Watching sport, I watch the bench as much as the play, because disappointment lives there.",
    "I notice who thanks the interpreter, the waiter, the person who fetched the chairs.",
    "A good day for me is one in which someone felt accurately seen.",
    "I would choose a messy human motive over a clean mechanical explanation when both are on offer.",
]

# Systemizing-direction narratives: attending to rules, mechanisms, taxonomies,
# and if-then regularities.
S_NARRATIVES: list[str] = [
    "When a kettle clicks off, I think about the bimetallic strip and the current path before I think about who wants tea.",
    "I sort tools by the job they do, then by the tolerance they need, then by how they fail.",
    "If a bicycle chain skips, I check wear, alignment, and cable tension in that order.",
    "I like maps that show how a system is allowed to change, not just where things sit.",
    "A recipe is a procedure. I want the temperatures, the times, and what happens if I invert two steps.",
    "When a spreadsheet breaks, I hunt the first cell whose assumption is false.",
    "I notice repeating motifs in machinery: gears trading torque for speed, valves trading flow for pressure.",
    "If a door sticks, I measure the gap and the humidity before I blame the person who closed it.",
    "I would rather have the state diagram than the anecdote.",
    "Watching rain on a window, I track how droplets merge and the path they take, not the mood of the weather.",
    "I build taxonomies for hobbies. The pleasure is in the clean split between kinds.",
    "When code fails, I reduce the input until the failure is a single rule I can name.",
    "I like clocks because the whole device is a contract about intervals.",
    "A traffic light is interesting as a protocol: who yields, who is timed, what happens when a sensor dies.",
    "I take things apart to see the hidden coupling, then I put them back so the coupling still holds.",
    "If two explanations compete, I keep the one that predicts the next measurement.",
    "I remember the exact sequence that made a printer jam, and I do not vary it blindly.",
    "In a new building I look for the service risers and the return path of the air.",
    "I am pulled toward diagrams more than toward faces. A diagram tells me what must be true.",
    "When music plays, I follow the time signature and the modulation more than the lyric's feeling.",
    "I want the interface contract first. Tone can wait.",
    "If a plant wilts, I check water, light, pot size, and salt before I invent a story about neglect.",
    "I enjoy a well-specified game: legal moves, terminal states, no hidden social scoring.",
    "A delayed reply makes me wonder about queues and retry policy before I wonder about mood.",
    "I am interested in how a bridge carries a load into the ground.",
    "When a stranger holds a door, I notice the closer's damping more often than the social bid.",
    "I keep a private catalogue of failure modes: what breaks first, and what that breakage reveals.",
    "If a story has a machine, I ask how the machine is powered and what it cannot do.",
    "I change my walking pace to the pavement grid and the timing of the crossings.",
    "Hearing a raised voice through a wall, I think about resonance and thin plaster before I think about a quarrel.",
    "I like conversations where we name the variables and then decide which to hold fixed.",
    "When someone over-explains a gadget, I listen for the causal chain and drop the extra adjectives.",
    "I remember how a mechanism felt when it finally seated longer than I remember the chat around it.",
    "I try to leave a problem with a rule that will still work tomorrow.",
    "If a device goes quiet, I check power, then input, then output, then the hidden interlock.",
    "I am curious about the moment a system leaves its designed operating range.",
    "Watching sport, I watch the formations and the constraint the offside line imposes.",
    "I notice who labelled the cables, numbered the boxes, and wrote the recovery steps.",
    "A good day for me is one in which a messy process became a rule I can reuse.",
    "I would choose a clean mechanical explanation over a messy human motive when both are on offer.",
]

assert len(E_NARRATIVES) >= 40
assert len(S_NARRATIVES) >= 40
assert len(set(E_NARRATIVES)) == len(E_NARRATIVES)
assert len(set(S_NARRATIVES)) == len(S_NARRATIVES)
