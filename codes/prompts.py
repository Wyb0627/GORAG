GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract or generate the following information:
- entity_name: Name of the entity, capitalized
- entity_description: Comprehensive description of the entity's attributes and activities, if there is no such description in the text, generate a description based on the entity's name and context.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}is the chair of{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}owned by{tuple_delimiter}5)
("relationship"{tuple_delimiter}CS{tuple_delimiter}CS is a.........)
{completion_delimiter}

######################
Example 3:
Text:
Acoustic stimulation refers to the use of sound waves to stimulate the brain, nervous system, or other parts of the body. It is a non-invasive technique that has been used in various fields, including medicine, psychology, and neuroscience.

Acoustic stimulation can take many forms, including:

Sound therapy: Using specific sounds or music to stimulate the brain and promote relaxation, reduce stress, or improve mood.
Auditory brain stimulation: Using sound to stimulate the brain's auditory system and improve cognitive function, such as attention, memory, or language processing.
Transcranial magnetic stimulation (TMS) with acoustic stimulation: Using sound waves to enhance the effects of TMS, a non-invasive brain stimulation technique that uses magnetic fields to stimulate brain activity.
Acoustic neuromodulation: Using sound waves to modulate brain activity, particularly in the treatment of neurological disorders such as epilepsy or Parkinson's disease.
Sound-induced brain stimulation: Using sound to stimulate the brain's default mode network, which is involved in introspection, self-reflection, and mind-wandering.
######################
Output:
("entity"{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}Acoustic stimulation refers to the use of sound waves to stimulate the brain, nervous system, or other parts of the body.)
{record_delimiter}
("entity"{tuple_delimiter}SOUND THERAPY{tuple_delimiter}Using specific sounds or music to stimulate the brain and promote relaxation, reduce stress, or improve mood.)
{record_delimiter}
("entity"{tuple_delimiter}AUDITORY BRAIN STIMULATION{tuple_delimiter}Using sound to stimulate the brain's auditory system and improve cognitive function.)
{record_delimiter}
("entity"{tuple_delimiter}COGNITIVE FUNCTION{tuple_delimiter}Cognitive function refers to the mental processes that allow us to process information, perceive, attend, remember, learn, reason, and solve problems.)
{record_delimiter}
("entity"{tuple_delimiter}TRANSCRANIAL MAGNETIC STIMULATION{tuple_delimiter} Using sound waves to enhance the effects of TMS, a non-invasive brain stimulation technique that uses magnetic fields to stimulate brain activity.)
{record_delimiter}
("entity"{tuple_delimiter}ACOUSTIC NEUROMODULATION{tuple_delimiter}Using sound waves to modulate brain activity, particularly in the treatment of neurological disorders such as epilepsy or Parkinson's disease.)
{record_delimiter}
("entity"{tuple_delimiter}SOUND-INDUCED BRAIN STIMULATION{tuple_delimiter}Using sound to stimulate the brain's default mode network, which is involved in introspection, self-reflection, and mind-wandering.)
{record_delimiter}
("entity"{tuple_delimiter}INTROSPECTION{tuple_delimiter}Introspection is the process of examining and observing one's own thoughts, feelings, and behaviors.)
{record_delimiter}
("entity"{tuple_delimiter}SELF-REFLECTION{tuple_delimiter}Self-reflection is the process of examining and evaluating one's own thoughts, feelings, and behaviors.)
{record_delimiter}
("entity"{tuple_delimiter}MIND-WANDERING{tuple_delimiter}Mind-wandering is the process of letting one's mind drift away from the present moment and task at hand, and engaging in internal mental activities such as daydreaming, ruminating, or fantasizing.)
{record_delimiter}
("relationship"{tuple_delimiter}SOUND THERAPY{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}form of{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}AUDITORY BRAIN STIMULATION{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}form of{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}TRANSCRANIAL MAGNETIC STIMULATION{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}form of{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}ACOUSTIC NEUROMODULATION{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}form of{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}SOUND-INDUCED BRAIN STIMULATION{tuple_delimiter}ACOUSTIC STIMULATION{tuple_delimiter}form of{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}AUDITORY BRAIN STIMULATION{tuple_delimiter}COGNITIVE FUNCTION{tuple_delimiter}improve{tuple_delimiter}7)
{completion_delimiter}

######################
-Real Data-
######################
Text: {input_text}
######################
Output:"""

TEXT_CLASSIFICATION_PROMPT = """
-Goal-
Given a text document, classify it into the most appropriate category among a given list of category labels. 

-Content-
You will be given the following content for text classification:

1. Text: The text for you to classify.

2. Category Labels: A list of category labels to choose from. You need to output exactly one category label from these labels.

3. Graph Path by Keywords: Paths between keyword entities extracted from the text on the pre-constructed graph. Different paths will be splited by {record_delimiter}. If a label from the category labels exists in these paths, it is likely to be an appropriate category label of the text. If there is no connected path between the keywords on the graph, this part will be empty.
Each path is formulated by entities of graph, for example, a length 2 path can be represented in such format:
(<source_entity_name>{path_delimiter}<entity1_name>{path_delimiter}<target_entity_name>)

Notations:
    -<source_entity_name>: the name of the source (beginning) entity in the path
    -<target_entity_name>: the name of the target (ending) entity in the path
    -<entity1_name>: the name of the first entity in the path
    -{path_delimiter}: delimiter to split the elements in the path

4. KG Subgraph by Keywords: Subgraph of the Knowledge Graph (KG) around the keywords extracted from the text. If all keywords of the text are not matched to KG, this part will be empty.
The KG subgraph is represented by triples of entities and relations in KG. Different triples will be splited by {record_delimiter}, for example:
(<entity_name>{tuple_delimiter}<relation_name>{tuple_delimiter}<entity_name>)
{record_delimiter}
(<entity_name>{tuple_delimiter}<relation_name>{tuple_delimiter}<entity_name>)

Notations:
    -<entity_name>: the name of the entity in the triple
    -<relation_name>: the name of the relation in the triple
    -{tuple_delimiter}: delimiter to split the elements in the triple
    
    
######################
-Examples-
######################
Example 1:
Text: 
Studies on the process of acoustic stimulation in prenatal period require to assess the beneficial condition for child development, as well as characteristic sounds which could negatively affect the development of auditory system, brain or even entire body. 

Category Labels: 
computer science, medical, civil, electrical and computer engineering, biochemistry, mechanical and aerospace engineering, psychology

KG Path by Keywords: 
(acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)
{record_delimiter}
(acoustic stimulation{path_delimiter}require to access{path_delimiter}child development)
{record_delimiter}
(acoustic stimulation{path_delimiter}belong to{path_delimiter}prenatal development)
{record_delimiter}
(acoustic stimulation{path_delimiter}belong to{path_delimiter}psychology)
{record_delimiter}
(child development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}belong to{path_delimiter}prenatal development)
{record_delimiter}
(child development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)
{record_delimiter}
(prenatal development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)

KG Subgraph by Keywords:
(acoustic stimulation{tuple_delimiter}related to{tuple_delimiter}prenatal development)
{record_delimiter}
(acoustic stimulation{tuple_delimiter}belong to{tuple_delimiter}psychology)
{record_delimiter}
(acoustic stimulation{tuple_delimiter}in{tuple_delimiter}prenatal period)
{record_delimiter}
(child development{tuple_delimiter}related to{tuple_delimiter}acoustic stimulation)
{record_delimiter}
(prenatal development{tuple_delimiter}related to{tuple_delimiter}psychology)
{record_delimiter}
(auditory system{tuple_delimiter}related to{tuple_delimiter}acoustic stimulation)

######################
Output: 
psychology


######################
-Real Data-
######################
Text: {input_text}

Category Labels: {labels}

KG Path by Keywords: {kg_path}

KG Subgraph by Keywords: {subgraph}
######################
Output:"""

CLASSIFICATION_PROMPT = """
-Goal-
Given information related to a unlabeled text, try to classify the text into the most appropriate category among a given list of category labels. 

-Content-
You will be given the following content for classification:

1. Category Labels: A list of category labels to choose from. You need to output exactly one category label from these labels.

3. Graph Path by Keywords: Paths between keyword entities extracted from the text on the pre-constructed graph. Different paths will be splited by {record_delimiter}. If a label from the category labels exists in these paths, it is likely to be an appropriate category label of the text. If there is no connected path between the keywords on the graph, this part will be empty.
Each path is formulated by entities of graph, for example, a length 2 path can be represented in such format:
(<source_entity_name>{path_delimiter}<entity1_name>{path_delimiter}<target_entity_name>)

Notations:
    -<source_entity_name>: the name of the source (beginning) entity in the path
    -<target_entity_name>: the name of the target (ending) entity in the path
    -<entity1_name>: the name of the first entity in the path
    -{path_delimiter}: delimiter to split the elements in the path

3. KG Subgraph by Keywords: Subgraph of the KG around the keywords extracted from the text. If all keywords of the text are not matched to KG, this part will be empty.
The KG subgraph is represented by triples of entities and relations in KG. Different triples will be splited by {record_delimiter}, for example:
(<entity_name>{tuple_delimiter}<relation_name>{tuple_delimiter}<entity_name>)
{record_delimiter}
(<entity_name>{tuple_delimiter}<relation_name>{tuple_delimiter}<entity_name>)

Notations:
    -<entity_name>: the name of the entity in the triple
    -<relation_name>: the name of the relation in the triple
    -{tuple_delimiter}: delimiter to split the elements in the triple


######################
-Examples-
######################
Example 1:
Category Labels: 
computer science, medical, civil, electrical and computer engineering, biochemistry, mechanical and aerospace engineering, psychology

KG Path by Keywords: 
(acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)
{record_delimiter}
(acoustic stimulation{path_delimiter}require to access{path_delimiter}child development)
{record_delimiter}
(acoustic stimulation{path_delimiter}belong to{path_delimiter}prenatal development)
{record_delimiter}
(acoustic stimulation{path_delimiter}belong to{path_delimiter}psychology)
{record_delimiter}
(child development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}belong to{path_delimiter}prenatal development)
{record_delimiter}
(child development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)
{record_delimiter}
(prenatal development{path_delimiter}related to{path_delimiter}acoustic stimulation{path_delimiter}in{path_delimiter}prenatal period)

KG Subgraph by Keywords:
(acoustic stimulation{tuple_delimiter}related to{tuple_delimiter}prenatal development)
{record_delimiter}
(acoustic stimulation{tuple_delimiter}belong to{tuple_delimiter}psychology)
{record_delimiter}
(acoustic stimulation{tuple_delimiter}in{tuple_delimiter}prenatal period)
{record_delimiter}
(child development{tuple_delimiter}related to{tuple_delimiter}acoustic stimulation)
{record_delimiter}
(prenatal development{tuple_delimiter}related to{tuple_delimiter}psychology)
{record_delimiter}
(auditory system{tuple_delimiter}related to{tuple_delimiter}acoustic stimulation)

######################
Output: 
psychology


######################
-Real Data-
######################
Category Labels: {labels}

KG Path by Keywords: {kg_path}

KG Subgraph by Keywords: {subgraph}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed.  Answer YES | NO if there are still entities or relationships that need to be added.\n"

FUZZY_MATCHING_PROMPT = """
-Goal-
Given a list of keywords and a list of target keywords, find no more than {max_linkage} keywords in the list that are most related to the target keywords.
If you think even there is no keywords related, you can return "None".

-Input-
The input will contain the following information:
1. A list of keywords.
2. A list of target keywords.

-Output-
The output should only contain the most related entity name from the list to the target keyword name, split by {tuple_delimiter}, or None if you think none of the provided entity is related.

######################
-Examples-
######################
Example 1:
Keywords: 
Food Company, Tech Company, American Company, University, Bank

Target Keywords: 
Apple Inc., Google, META, Facebook, ORACLE

######################
Output:
Tech Company, American Company

######################
-Real Data-
######################
Keywords: 
{entity_names}

Target Keywords: 
{target_entity_names}
######################
Output:"""

system_prompt_graph_filter_step = """
Given a list of graph nodes and a list of nodes to be added to the graph, your task is to choose a subset of nodes within the nodes to be added 
that can complement the existing graph nodes and filter out those that already have similar nodes within the graph. 
You need to respond only with the filtered nodes, separated by commas; do not respond with anything else. 
"""

system_prompt_graph_filter = """
Given a list of graph nodes, your task is to coarse the graph, reduce the number of nodes, filter out those that already have similar nodes within the graph, 
and make sure the important terminal nodes within another given list of nodes are not filtered.\n
You need to respond with the one-to-one mapping of the filtered nodes and the most similar node to them that are not filtered, 
in a JSON string of a Python dict, do not respond with anything else, and make sure the nodes within your output are in the graph.\n
An example output format is given below:\n
{'filtered_node1':'similar_node1','filtered_node2':'similar_node2','filtered_node3':'similar_node3'}
"""