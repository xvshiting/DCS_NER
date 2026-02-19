this is an repo for LDG-Gliner paper. 
This approach benifit from llm and Span extraction model Gliner.
We train an LLM to generate label and label description based on text.  To contruct an ner dataset with variety label and label description. Then train an Gliner model it can performe better based on those information ,and could generailize to other label and label description.  Then we fix Gliner and use GRPO to train LLM, force the LLM to generate more helful description. 

##  dataset 
- pile ner type: 
   -path "/data/dataset/ner/Pile_NER_type"
   load use datasets load_dataset function.

- pile ner definition: 
   -path "/data/dataset/ner/Pile_NER_definition"
   load use datasets load_dataset function.

