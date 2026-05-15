PROMPT3_context = """
"""

PROMPT1_entity_image_0_image = """You are tasked with describing an entity based on the following details:

- Entity Name: {entity_name}
- Attributes: {attr}
- Instance Type: {instance}
- Description: {desc}

Answer follow the format: "The {entity_name} refer to..."
Describe the entity and directly provide the final description result."""

PROMPT1_entity_image_1n_image = """Your task is to describe the entity {entity_name} in images based on the following information.

- Entity Name: {entity_name}
- Attributes: {attr}
- Instance Type: {instance}
- Description: {desc}

Answer follow the format: "The {entity_name} refer to..."
Only generate an introduction to the target entity in images, not a description of images, directly provide the final description result."""

PROMPT1_entity_image_text_summary = """Please generate a summary for the given entity, including entity name and description. Based on the following information.
- Entity Name: {entity_name}
- Attributes: {attr}
- Instance Type: {instance}
- Description: {desc}
- Description of images related to the entity: {desc_image}
Try your best to summarize the main content of the given entity based on the above information. And generate a summary for it, only output the summary.
Summary:"""

PROMPT1_entity_image_text_1sent = """Please generate a one-sentence summary for the given entity, including entity name and description. Based on the following information.
- Entity Name: {entity_name}
- Attributes: {attr}
- Instance Type: {instance}
- Description: {desc}
- Description of images related to the entity: {desc_image}
Try your best to summarize the main content of the given entity based on the above information. And generate a short summary in 1 sentence for it, only output the summary.
Summary:"""

PROMPT2_mention_image_no_image = """The target entity is named {mentions}.
The sentence related to the entity is {sentence}.
Introduce the entity {mentions}. Answer follow the format: "The {mentions} refer to..."
Only output the introduction."""

PROMPT2_mention_image_1n_image = """Your task is to describe an entity based on the provided images, and the image descriptions.
The target entity is named {mentions}.
The description of image related to the entity is {sentence}.
Introduce the entity {mentions}. Answer follow the format: "The {mentions} refer to..."
Only generate an introduction to the target entity based on image and description, not a description of the image.
Only output the introduction."""

PROMPT2_mention_image_text_summary = """Please generate a summary for the given entity, including entity name and description. Based on the following information.
- Entity Name: {mentions}
- sentence: {sentence}
- Description of images related to the entity: {desc_image}
Try your best to summarize the main content of the given entity based on the above information. And generate a summary for it, only output the summary.
Summary:"""

PROMPT2_mention_image_text_1sent = """Please generate a one-sentence summary for the given entity, including entity name and description. Based on the following information.
- Entity Name: {mentions}
- sentence: {sentence}
- Description of images related to the entity: {desc_image}
Try your best to summarize the main content of the given entity based on the above information. And generate a short summary in 1 sentence for it, only output the summary.
Summary:"""

PROMPT3_knowledge_contrast_1to30 = """Based on the following entity table, your task is to describe the differences between Entity {Entity_Num} [{Entity_Name}] and the other entities based on this information.

### Entity table
{Entity_table_info}

### Output Format
Unlike the other entities, {Entity_Name} is [distinctive characteristic based on its category, description, or unique features].

output only the final answer without any additional information.
"""

PROMPT3_knowledge_contrast2 = """Based on the following entity table, generate a list where each list item describes how that entity is different from the others. The list should have the same number of elements as there are entities in the table. Each element should follow this format:
### Entity table
{Entity_table_info}

### Output Format
1. Unlike the other entities, [Entity Name] is [distinctive characteristic based on its category, description, or unique features].
2.
...
30.

Ensure that each description highlights the entity's unique aspects compared to all other entities, and output only the list without any additional information.
"""

PROMPT3_knowledge_contrast1 = """Below is a list of entities along with their related descriptions.
Your task is to identify their differences, and your description should highlight what sets each entity apart from the others.

### Entity table
{Entity_table_info}

Output a JSON following the format:
[entity_qid: difference_description, entity_qid: difference_description,...].
Only output a json, not other inforamation."""

PROMPT3_ranker = """You are an expert in knowledge graph and entity matching, specifically ranking candidate entities by relevance.

### Task
You will be given:
1. A mention with name, context, description, and category.
2. An entity table containing candidate entities.
3. If the name of the mention is exactly the same as the name of the entity, this is usually the best match.

### Mention
Name: {mention_name}
Context: {mention_context}
Description: {mention_des}
Category: {mention_cate}

### Entity table
{Entity_table_info}

### Output Format
Rank all 30 entities in the Entity Table based on their relevance to the given mention. Output a space-separated list of entity IDs in descending order of relevance. The most relevant entity should appear first.

### Output (Example)
"1 8 11 12 21 22 13 5 26 23 24 15 30 14 29 3 28 20 27 9 19 25 2 6 18 7 17 16 10 4"

### Rules
- Strictly follow the output format: a single line of numbers, separated by spaces.
- Do not include explanations or extra text.
- If the format is incorrect, the answer is considered invalid.
- Sort by the most relevant entity first."""

PROMPT3_multi_choice_1 = """You are an expert in knowledge graph, and matching at top k specifically.
Given the following mention and entity table, identify the most relevant entity from the entity table and output ONLY the corresponding entity number. Do not provide any explanation or additional text.

### Mention
Name: {mention_name}
Context: {mention_context}
Description: {mention_des}
Category: {mention_cate}

### Entity table
{Entity_table_info}

### Output format:
{OutPut}"""

PROMPT3_multi_choice_many = """You are an expert in knowledge graph, and matching at top k specifically.
Given the following mention and entity table, identify the 20 most relevant entities from the entity table and output ONLY their corresponding entity numbers in descending order of relevance. Do not provide any explanation or additional text.

### Mention
Name: {mention_name}
Context: {mention_context}
Description: {mention_des}
Category: {mention_cate}

### Entity table
{Entity_table_info}

### Output format:
{OutPut}"""

PROMPTS = {
    "PROMPT1_entity_image_no_image": PROMPT1_entity_image_0_image,
    "PROMPT1_entity_image_1n_image": PROMPT1_entity_image_1n_image,
    "PROMPT1_entity_image_text_summary": PROMPT1_entity_image_text_summary,
    "PROMPT1_entity_image_text_1sent": PROMPT1_entity_image_text_1sent,
    "PROMPT2_mention_image_no_image": PROMPT2_mention_image_no_image,
    "PROMPT2_mention_image_1n_image": PROMPT2_mention_image_1n_image,
    "PROMPT2_mention_image_text_summary": PROMPT2_mention_image_text_summary,
    "PROMPT2_mention_image_text_1sent": PROMPT2_mention_image_text_1sent,
    "PROMPT3_knowledge_contrast": PROMPT3_knowledge_contrast_1to30,
    "PROMPT3_multi_choice_1": PROMPT3_multi_choice_1,
    "PROMPT3_ranker": PROMPT3_ranker,
}
