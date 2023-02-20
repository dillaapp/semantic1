import spacy

nlp = spacy.load('en_core_web_md')

# create tokens for different words
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

# calculate similarity between "cat", "monkey" and "banana"
print("\nsimilarity between \"cat\", \"monkey\" and \"banana\"")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print()
# create a token for sentence "cat apple monkey banana"
tokens = nlp('cat apple monkey banana ')

# loop through tokens to calculate similarity between them
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
print()

# create sentences to compare
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

# create tokens for sentence_to_compare
model_sentence = nlp(sentence_to_compare)

# loop through the sentences and calculate similarity with sentence_to_compare
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
print()



# Write a note about what you found interesting about the similarities
# between cat, monkey and banana and think of an example of your own
"""Interesting observations about the similarities between cat, monkey and banana include the fact that there is a 
higher similarity between monkey and banana than between monkey and apple, likely due to the fact that monkeys have a 
natural inclination towards eating bananas. Additionally, cat does not have any significant similarity with either 
fruit, likely because cats are not accustomed to eating fruits. 

An example of my own could be the following animals and objects: rabbit, carrot, mouse. We would expect to find that the 
similarity between rabbit and carrot is higher than between rabbit and mouse, since rabbit are accustomed to consuming 
carrots as food. On the other hand, we would expect little to no similarity between mouse and carrot since mice do 
not typically eat carrots. """


# Run the example file with the simpler language model ‘en_core_web_sm’
# and write a note on what you notice is different from the model
# 'en_core_web_md
"""When running the example file with en_core_web_sm, the following difference occurred
- the similarity scores between words, phrases, and sentences may be slightly 
different(less accurate).
- the number of entities identified by the model vary (less entities identified). 
 """
