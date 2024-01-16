# Eval description
This eval takes a set of user criteria, a raw winelist content and the curated list of wines included in the raw wine list, and a set of examples. 
The expected response is a json. 
Using a pydantic model, it checks the following:
  - the response follows the same json schema as the provided examples
  - the suggested wines are present in the list
  - the explanation field is filled
  - wine color is picked in the set of allowed options

