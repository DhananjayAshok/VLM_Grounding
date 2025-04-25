IDENTIFICATION_EVALUATOR_CONTEXT = """
You are an impartial and logical evaluator tasked with analyzing two sentences and determining if the candidate has identified the same object as the reference. 
"""

IDENTIFICATION_EXAMPLE_1_Q = """
Candidate: The object in the image is a golf ball
Reference: Golf Ball
"""

IDENTIFICATION_EXAMPLE_1_A = """
Explanation: The candidate sentence identifies the object correct as a golf ball
Judgment: PASS [STOP]
"""

IDENTIFICATION_EXAMPLE_2_Q = """
Candidate: The image shows a digit, however it is not clear what the digit is
Reference: 6
"""

IDENTIFICATION_EXAMPLE_2_A = """
Explanation: The candidate sentence does not identify the object correctly as a 6
Judgment: FAIL [STOP]
"""

IDENTIFICATION_PROMPTS = [IDENTIFICATION_EVALUATOR_CONTEXT, (IDENTIFICATION_EXAMPLE_1_Q, IDENTIFICATION_EXAMPLE_1_A), (IDENTIFICATION_EXAMPLE_2_Q, IDENTIFICATION_EXAMPLE_2_A)]

CORRECTNESS_EVALUATOR_CONTEXT = """
You are an impartial and logical evaluator tasked with analyzing two sentences and determining if the candidate has the same overall meaning as the reference. 
"""

CORRECTNESS_EXAMPLE_1_Q = """
Candidate: The ball is made of rubber
Reference: Rubber
"""

CORRECTNESS_EXAMPLE_1_A = """
Explanation: The candidate sentence identifies rubber as the material of the ball
Judgment: PASS [STOP]
"""

CORRECTNESS_EXAMPLE_2_Q = """
Candidate: Cold and dark water
Reference: Freshwater
"""

CORRECTNESS_EXAMPLE_2_A = """
Explanation: The candidate sentence identifies cold and dark water, which is different from freshwater
Judgment: FAIL [STOP]
"""

CORRECTNESS_PROMPTS = [CORRECTNESS_EVALUATOR_CONTEXT, (CORRECTNESS_EXAMPLE_1_Q, CORRECTNESS_EXAMPLE_1_A), (CORRECTNESS_EXAMPLE_2_Q, CORRECTNESS_EXAMPLE_2_A)]

QUESTION_EXTRACTION_CONTEXT = """
You are a logical system tasked with extracting questions for an entity from a given text. The question should have a unique answer, and should be very short.
"""

QUESTION_EXTRACTION_EXAMPLE_1_Q = """
Entity: Tench
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia as far as the Ob and Yenisei Rivers. It is also found in Lake Baikal. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers.
"""

QUESTION_EXTRACTION_EXAMPLE_1_A = """
Rationale: The tench is said to also be called the doctor fish. There is no other alternate name. 
Question: What is another name for the tench?
Answer: doctor fish
[SEP]
Rationale: The order of the tench is Cypriniformes. Animals can only belong to one order.
Question: What is the order of the tench?
Answer: Cypriniformes
[SEP]
Rationale: The usual habbitat of the tench is freshwater. 
Question: What kind of water does the tench usually live in?
Answer: freshwater 
[STOP]
"""

QUESTION_EXTRACTION_EXAMPLE_2_Q = """
Entity: Baklava
Text: Baklava ( , or ; Ottoman Turkish: باقلوا) is a layered pastry dessert made of filo pastry, filled with chopped nuts, and sweetened with syrup or honey. It was one of the most popular sweet pastries of Ottoman cuisine.\nThere are several theories for the origin of the pre-Ottoman version of the dish. In modern times, it is a common dessert among cuisines of countries in West Asia, Southeast Europe, Central Asia, and North Africa. It is also enjoyed in Pakistan and Afghanistan, where, although not a traditional sweet, it has carved out a niche in urban centers.
"""

QUESTION_EXTRACTION_EXAMPLE_2_A = """
Rationale: Baklava is associated with Ottoman cuisine.
Question: What ancient cuisine is Baklava associated with?
Answer: Ottoman
[SEP]
Rationale: Baklava is made of filo pastry.
Question: What kind of pastry is Baklava made of?
Answer: filo
[STOP]
"""

QUESTION_EXTRACTION_EXAMPLE_3_Q = """
Entity: Eiffel Tower
Text: The Eiffel Tower (  EYE-fəl; French: Tour Eiffel [tuʁ ɛfɛl] ) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.\nLocally nicknamed "La dame de fer" (French for "Iron Lady"), it was constructed as the centerpiece of the 1889 World\'s Fair, and to crown the centennial anniversary of the French Revolution. Although initially criticised by some of France\'s leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world. The tower received 5,889,000 visitors in 2022. The Eiffel Tower is the most visited monument with an entrance fee in the world: 6.91 million people ascended it in 2015. It was designated a monument historique in 1964, and was named part of a UNESCO World Heritage Site ("Paris, Banks of the Seine") in 1991.\nThe tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
"""

QUESTION_EXTRACTION_EXAMPLE_3_A = """
Rationale: The Eiffel Tower is located in Paris.
Question: In which city is the Eiffel Tower located?
Answer: Paris
[SEP]
Rationale: The Eiffel Tower is made of wrought-iron.
Question: What material is the Eiffel Tower made of?
Answer: wrought-iron
[SEP]
Rationale: The Eiffel Tower was built to crown the centennial anniversary of the French Revolution.
Question: The Eiffel Tower was built to celebrate the centennial anniversary of which event?
Answer: French Revolution
[STOP]
"""

QUESTION_EXTRACTION_PROMPTS = [QUESTION_EXTRACTION_CONTEXT, (QUESTION_EXTRACTION_EXAMPLE_1_Q, QUESTION_EXTRACTION_EXAMPLE_1_A), (QUESTION_EXTRACTION_EXAMPLE_2_Q, QUESTION_EXTRACTION_EXAMPLE_2_A), (QUESTION_EXTRACTION_EXAMPLE_3_Q, QUESTION_EXTRACTION_EXAMPLE_3_A)]

QUESTION_UNIQUE_ANSWER_CONTEXT = """
Given a text and a question, judge whether the question has a unique answer, or can be answered with multiple valid responses.
"""

QUESTION_UNIQUE_ANSWER_EXAMPLE_1_Q = """
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia as far as the Ob and Yenisei Rivers. It is also found in Lake Baikal. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers.
Question: What is the tench also known as?
"""

QUESTION_UNIQUE_ANSWER_EXAMPLE_1_A = """
Rationale: The text mentions only one other name for the tench, which is doctor fish.
Judgment: Unique [STOP]
"""

QUESTION_UNIQUE_ANSWER_EXAMPLE_1_Q = """
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia as far as the Ob and Yenisei Rivers. It is also found in Lake Baikal. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers.
Question: Which lake is the tench found in? 
"""

QUESTION_UNIQUE_ANSWER_EXAMPLE_1_A = """
Rationale: The text mentions that the tench is found in Lake Baikal. However, it is very likely that the tench is found in other lakes as well.
Judgment: Multiple [STOP]
"""

QUESTION_UNIQUE_ANSWER_PROMPTS = [QUESTION_UNIQUE_ANSWER_CONTEXT, (QUESTION_UNIQUE_ANSWER_EXAMPLE_1_Q, QUESTION_UNIQUE_ANSWER_EXAMPLE_1_A)]


QUESTION_ANSWER_CONTEXT = """
Given a question and a context, provide a short answer to the question based on the information in the context.
"""

QUESTION_ANSWER_CONTEXT_EXAMPLE_1_Q = """
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia as far as the Ob and Yenisei Rivers. It is also found in Lake Baikal. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers.
Question: What is the tench also known as?
"""

QUESTION_ANSWER_CONTEXT_EXAMPLE_1_A = """
Answer: doctor fish [STOP]
"""

QUESTION_ANSWER_CONTEXT_EXAMPLE_2_Q = """
Text: Baklava ( , or ; Ottoman Turkish: باقلوا) is a layered pastry dessert made of filo pastry, filled with chopped nuts, and sweetened with syrup or honey. It was one of the most popular sweet pastries of Ottoman cuisine.\nThere are several theories for the origin of the pre-Ottoman version of the dish. In modern times, it is a common dessert among cuisines of countries in West Asia, Southeast Europe, Central Asia, and North Africa. It is also enjoyed in Pakistan and Afghanistan, where, although not a traditional sweet, it has carved out a niche in urban centers.
Question: What kind of pastry is Baklava made of?
"""

QUESTION_ANSWER_CONTEXT_EXAMPLE_2_A = """
Answer: filo [STOP]
"""

QUESTION_ANSWER_CONTEXT_PROMPTS = [QUESTION_ANSWER_CONTEXT, (QUESTION_ANSWER_CONTEXT_EXAMPLE_1_Q, QUESTION_ANSWER_CONTEXT_EXAMPLE_1_A), (QUESTION_ANSWER_CONTEXT_EXAMPLE_2_Q, QUESTION_ANSWER_CONTEXT_EXAMPLE_2_A)]


QUESTION_ANSWER = """
You are a knowledgeable system tasked with answering the question based on your knowledge.
"""

QUESTION_ANSWER_EXAMPLE_1_Q = """
Question: What is the tench also known as?
"""

QUESTION_ANSWER_EXAMPLE_1_A = """
Answer: doctor fish [STOP]
"""

QUESTION_ANSWER_EXAMPLE_2_Q = """
Question: What kind of pastry is Baklava made of?
"""

QUESTION_ANSWER_EXAMPLE_2_A = """
Answer: filo [STOP]
"""

QUESTION_ANSWER_EXAMPLE_3_Q = """
Question: In which city is the Eiffel Tower located?
"""
QUESTION_ANSWER_EXAMPLE_3_A = """
Answer: Paris [STOP]
"""

QUESTION_ANSWER_PROMPTS = [QUESTION_ANSWER, (QUESTION_ANSWER_EXAMPLE_1_Q, QUESTION_ANSWER_EXAMPLE_1_A), (QUESTION_ANSWER_EXAMPLE_2_Q, QUESTION_ANSWER_EXAMPLE_2_A), (QUESTION_ANSWER_EXAMPLE_3_Q, QUESTION_ANSWER_EXAMPLE_3_A)]


QUESTION_EXTRACTION_MCQ = """
You are a logical system tasked with generating incorrect options for multiple choice questions. You are given the text, question and answer. Come up with a numbered list of three plausible but incorrect options
"""

QUESTION_EXTRACTION_MCQ_EXAMPLE_1_Q = """
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia as far as the Ob and Yenisei Rivers. It is also found in Lake Baikal. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers.
Question: What is the tench also known as?
Correct Answer: doctor fish
"""

QUESTION_EXTRACTION_MCQ_EXAMPLE_1_A = """
Incorrect Option 1: miracle fish
Incorrect Option 2: salmon 
Incorrect Option 3: hidden fish
[STOP]
"""

QUESTION_EXTRACTION_MCQ_EXAMPLE_2_Q = """
Text: Baklava ( , or ; Ottoman Turkish: باقلوا) is a layered pastry dessert made of filo pastry, filled with chopped nuts, and sweetened with syrup or honey. It was one of the most popular sweet pastries of Ottoman cuisine.\nThere are several theories for the origin of the pre-Ottoman version of the dish. In modern times, it is a common dessert among cuisines of countries in West Asia, Southeast Europe, Central Asia, and North Africa. It is also enjoyed in Pakistan and Afghanistan, where, although not a traditional sweet, it has carved out a niche in urban centers.
Question: What kind of pastry is Baklava made of?
Correct Answer: filo
"""

QUESTION_EXTRACTION_MCQ_EXAMPLE_2_A = """
Incorrect Option 1: puff
Incorrect Option 2: shortcrust
Incorrect Option 3: choux
[STOP]
"""

QUESTION_EXTRACTION_MCQ_EXAMPLE_3_Q = """
Text: The Eiffel Tower (  EYE-fəl; French: Tour Eiffel [tuʁ ɛfɛl] ) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.\nLocally nicknamed "La dame de fer" (French for "Iron Lady"), it was constructed as the centerpiece of the 1889 World\'s Fair, and to crown the centennial anniversary of the French Revolution. Although initially criticised by some of France\'s leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world. The tower received 5,889,000 visitors in 2022. The Eiffel Tower is the most visited monument with an entrance fee in the world: 6.91 million people ascended it in 2015. It was designated a monument historique in 1964, and was named part of a UNESCO World Heritage Site ("Paris, Banks of the Seine") in 1991.\nThe tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
Question: In which city is the Eiffel Tower located?
Correct Answer: Paris
"""
QUESTION_EXTRACTION_MCQ_EXAMPLE_3_A = """
Incorrect Option 1: London
Incorrect Option 2: Berlin
Incorrect Option 3: Rome
[STOP]
"""

QUESTION_EXTRACTION_MCQ_PROMPTS = [QUESTION_EXTRACTION_MCQ, (QUESTION_EXTRACTION_MCQ_EXAMPLE_1_Q, QUESTION_EXTRACTION_MCQ_EXAMPLE_1_A), (QUESTION_EXTRACTION_MCQ_EXAMPLE_2_Q, QUESTION_EXTRACTION_MCQ_EXAMPLE_2_A), (QUESTION_EXTRACTION_MCQ_EXAMPLE_3_Q, QUESTION_EXTRACTION_MCQ_EXAMPLE_3_A)]

QUESTION_ANSWER_MCQ = """
You are a knowledgeable system tasked with answering the question based on your knowledge. The answer should be in the multiple choice format with 4 options. Output both incorrect and correct options for the question
"""

QUESTION_ANSWER_MCQ_EXAMPLE_1_Q = """
Question: What is the tench also known as?
Option 1: doctor fish
Option 2: miracle fish
Option 3: salmon
Option 4: hidden fish
"""
QUESTION_ANSWER_MCQ_EXAMPLE_1_A = """
Answer: Option 1: doctor fish [STOP]
"""

QUESTION_ANSWER_MCQ_EXAMPLE_2_Q = """
Question: What kind of pastry is Baklava made of?
Option 1: shortcrust
Option 2: puff
Option 3: filo
Option 4: choux
"""
QUESTION_ANSWER_MCQ_EXAMPLE_2_A = """
Answer: Option 3: filo [STOP]
"""

QUESTION_ANSWER_MCQ_EXAMPLE_3_Q = """
Question: In which city is the Eiffel Tower located?
Option 1: London
Option 2: Paris
Option 3: Berlin
Option 4: Rome
"""
QUESTION_ANSWER_MCQ_EXAMPLE_3_A = """
Answer: Option 2: Paris [STOP]
"""
QUESTION_ANSWER_MCQ_PROMPTS = [QUESTION_ANSWER_MCQ, (QUESTION_ANSWER_MCQ_EXAMPLE_1_Q, QUESTION_ANSWER_MCQ_EXAMPLE_1_A), (QUESTION_ANSWER_MCQ_EXAMPLE_2_Q, QUESTION_ANSWER_MCQ_EXAMPLE_2_A), (QUESTION_ANSWER_MCQ_EXAMPLE_3_Q, QUESTION_ANSWER_MCQ_EXAMPLE_3_A)]


QA_DUPLICATE = """
You are a logical system tasked with determining if two question answer pairs are duplicates of each other.
"""

QA_DUPLICATE_EXAMPLE_1_Q = """
Question: What is the tench also known as?
Answer: doctor fish
Question: What is another name for the tench?
Answer: the doctor fish
"""
QA_DUPLICATE_EXAMPLE_1_A = """
Rationale: The two questions are asking the same thing, and have the same answer.
Judgment: Duplicate [STOP]
"""

QA_DUPLICATE_EXAMPLE_2_Q = """
Question: What pastry is Baklava made of?
Answer: filo
Question: What kind of nuts are used to make Baklava?
Answer: walnuts
"""

QA_DUPLICATE_EXAMPLE_2_A = """
Rationale: The two questions are asking about different things, and have different answers.
Judgment: Unique [STOP]
"""

QA_DUPLICATE_PROMPTS = [QA_DUPLICATE, (QA_DUPLICATE_EXAMPLE_1_Q, QA_DUPLICATE_EXAMPLE_1_A), (QA_DUPLICATE_EXAMPLE_2_Q, QA_DUPLICATE_EXAMPLE_2_A)]


WIKI_GENERATION = """
You are a reliable and knowledgeable system tasked with generating an exhaustive text article with facts about a given entity. The article should be informative and concise.
"""

WIKI_GENERATION_EXAMPLE_1_Q = """
Entity: Tench
"""

WIKI_GENERATION_EXAMPLE_1_A = """
Text: The tench or doctor fish (Tinca tinca) is a fresh- and brackish-water fish of the order Cypriniformes found throughout Eurasia from Western Europe including Britain and Ireland east into Asia. It normally inhabits slow-moving freshwater habitats, particularly lakes and lowland rivers. The tench was first formally described in as Cyprinus tinca by Carl Linnaeus in 1758. The tench is most often found in still waters with a clay or muddy substrate and abundant vegetation. This species is rare in clear waters across stony substrate, and is absent altogether from fast-flowing streams. It tolerates water with a low oxygen concentration, being found in waters where even the carp cannot survive. Tench feed mostly at night with a preference for animals, such as chironomids, on the bottom of eutrophic waters and snails and pea clams in well-vegetated waters.
"""

WIKI_GENERATION_EXAMPLE_2_Q = """
Entity: Baklava
"""

WIKI_GENERATION_EXAMPLE_2_A = """
Text: Baklava is a layered pastry dessert made of filo pastry, filled with chopped nuts, and sweetened with syrup or honey. It was one of the most popular sweet pastries of Ottoman cuisine. There are claims attributing baklava to the Assyrians, according to which baklava was prepared by them in the 8th century BC. Many claim that the placenta cake, and therefore likely baklava, derived from a recipe from Ancient Greece. Baklava is usually served at room temperature, and is often garnished with nuts that have been ground up.
"""

WIKI_GENERATION_PROMPTS = [WIKI_GENERATION, (WIKI_GENERATION_EXAMPLE_1_Q, WIKI_GENERATION_EXAMPLE_1_A), (WIKI_GENERATION_EXAMPLE_2_Q, WIKI_GENERATION_EXAMPLE_2_A)]


prompts_dict = {
    "identification_evaluation": IDENTIFICATION_PROMPTS, 
    "correctness_evaluation": CORRECTNESS_PROMPTS, 
    "question_extraction": QUESTION_EXTRACTION_PROMPTS, 
    "question_extraction_mcq": QUESTION_EXTRACTION_MCQ_PROMPTS,
    "uniqueness_validation": QUESTION_UNIQUE_ANSWER_PROMPTS,
    "question_duplicate_evaluation": QA_DUPLICATE_PROMPTS,
    "question_answering": QUESTION_ANSWER_PROMPTS,
    "question_answering_mcq": QUESTION_ANSWER_MCQ_PROMPTS,
    "wiki_generation": WIKI_GENERATION_PROMPTS,
}