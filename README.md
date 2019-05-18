## COMP90042_Project_Automatic_Fact_Verification
###### This is the project of COMP90042 Web Search and Analysis. Group work by 2 master students.



#### Submission materials:

* Report (.pdf)
* Python code (.py or ipynb)
* Scripting code (.sh or similar), if using Unix command line tools for preprocessing

#### Datasets

You are provided with several data files for use in the project: 

`wiki-text.zip` a collection of wikipedia documents

`training.json` a set of training claims and answers

`devset.json` a set of development claims and answers

`test-unlabelled.json` a set of test claims (without answers) 

Most of datafiles are json, formulated as a dictionary of key-value pairs linking a claim identified with its details. E.g., 

```json
"75397": {
    "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
    "label": "SUPPORTS",
    "evidence": [
      ["Fox_Broadcasting_Company", 0],
      ["Nikolaj_Coster-Waldau", 7]
    ]
}
```

The entries under *evidence* denote the sentences in the collection which support or refute the claim (no evidence is provided for label=***NOT ENOUGH INFO***.)

#### Explaination

* **Label**
  * `SUPPORTED` : have evidence to support this claim
  * `REFUTED` : have evidence to refute this claim
  * `NOTENOUGHINFO` : no evidence is provided for this claim
* **Evidence** : Show the evidence location
  * `"Fox_Broadcasting_Company"` : show the page identifier
  * `0` : show the sentence index in that page