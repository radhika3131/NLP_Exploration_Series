# Chapter 1: Introduction to NLP And Transformers, what can they do?
Welcome to "NLP Exploration: Chapter by Chapter with Me," a series dedicated to unraveling the power of Transformer models in Natural Language Processing (NLP) and beyond. In this series, we'll embark on a journey through key concepts, practical techniques, and advanced applications using 🤗 Transformers from Hugging Face. I will post new chapters weekly so that we can grow and learn together along this journey.
## Understanding NLP
NLP is a field of Artificial Intelligence focused on understanding everything related to human language or focuses on the interaction between computers and humans through language. The goal of NLP is to enable computers to understand, interpret, and respond to human language in a way that is both meaningful and useful. Think of it as teaching computers to read, write, listen, and speak like humans. Here are some simple examples.
#### Reading and Understanding Text:
NLP allows computers to read and understand the text. For example, when you type a question into a search engine, NLP helps the computer understand what you're asking

##### Writing Text:
* NLP can help computers generate text. For example, when you use predictive text or autocomplete on your phone, NLP helps to predict and suggest the next word or phrase.

NLP isn't limited to written text though. It also tackles complex challenges in speech recognition and computer vision, such as generating a transcript of an audio sample or a description of an image.
#### Listening and Understanding Speech:
* NLP enables computers to understand spoken language. For instance, virtual assistants like Siri or Alexa can understand and respond to your voice commands because of NLP.

##### Speaking:
* NLP can also help computers generate spoken language. This is what happens when your GPS gives you directions or when text-to-speech software reads out text to you.

##### **Real-World Examples of NLP** :
* **Spam Filters**: Email services use NLP to filter out spam messages.
* **Customer Service**: Chatbots use NLP to help answer customer queries.
* **Voice Assistants**: Devices like Google Home, Siri, and Alexa use NLP to understand and respond to voice commands.
* **Language Translation**: Apps like Google Translate use NLP to translate text and speech between different languages.

##### **Why is it Challenging?**
computers don't process information in the same way as humans. For example, when we read the sentence "I am hungry," we can easily understand its meaning. Similarly, given two sentences such as "I am hungry" and "I am sad," we're able to easily determine how similar they are. For machine learning (ML) models, such tasks are more difficult. The text needs to be processed in a way that enables the model to learn from it. And because language is complex, we need to think carefully about how this processing must be done.
Language is full of nuances, ambiguity, and context dependence, making it challenging for computers to understand and generate it accurately. Different languages have varied grammatical structures, dialects, and idioms, further complicating the task. Sarcasm, irony, and slang add another layer of complexity.
![](https://github.com/radhika3131/NLP_Exploration_Series/blob/main/images/Art1im1.jpg)
#### Introduction to Transformers
This is where Transformers come into the picture. Transformers are a powerful type of model that has revolutionized NLP by improving how we process and understand text. They can handle the complexity of language more effectively than previous methods, allowing for better performance on tasks like translation, summarization, and sentiment analysis.
#### Transformers, What can they do?
Transformers have proven to be incredibly versatile and effective in various NLP tasks, driving advancements in AI and improving the way we interact with technology. Here are some key tasks that Transformers excel at, along with examples:
* **Machine Translation**: Transformers are highly effective in translating text from one language to another. For example, Google's Neural Machine Translation system uses Transformers to provide accurate and fluent translations. When translating the English sentence "The weather is nice today" into French, a Transformer model can produce "Il fait beau aujourd'hui."
* **Text Summarization**: Transformers can generate concise summaries of long documents. For instance, given a lengthy article about climate change, a Transformer-based model like BART can produce a brief summary highlighting the key points, such as "Global temperatures are rising due to increased greenhouse gas emissions, leading to severe weather patterns and melting ice caps."

Before diving into how Transformer models work, let's look at a few examples of how they can be used to solve some interesting NLP problems.
##### Working with Pipelines:
Hugging Face provides a powerful and user-friendly interface called pipelines, The most basic object in the 🤗 Transformers library is the pipeline() function. which allows you to quickly leverage Transformer models for various NLP tasks without delving into the complexities of the underlying architecture. It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer.

Here is how you can use these pipelines to solve interesting NLP problems:
**Sentiment Analysis:** Sentiment analysis determines the emotional tone behind a piece of text. Using a sentiment analysis pipeline, you can quickly classify text as positive, negative, or neutral.
```
from transformers import pipeline
# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Analyze the sentiment of a sentence
result = sentiment_analysis("I love using Transformer models!")
print(result)
```
output: [{'label': 'POSITIVE', 'score': 0.9995970129966736}]

By default, this pipeline selects a particular pretrained model that has been fine-tuned for sentiment analysis in English. The model is downloaded and cached when you create the sentiment_analysis object. If you rerun the command, the cached model will be used instead and there is no need to download the model again.

There are three main steps involved when you pass some text to a pipeline:

* The text is preprocessed into a format the model can understand.
* The preprocessed inputs are passed through the model to generate predictions. The model uses its learned weights and architecture to process the input data and produce output.
* The predictions of the model are post-processed, so you can make sense of them. The raw output from the model is converted into a human-readable format. This step involves interpreting the model's output and transforming it into a more understandable form.

Some of the currently *available pipelines* are:
* feature-extraction (get the vector representation of a text)
* fill-mask
* ner (named entity recognition)
* question-answering
* sentiment-analysis
* summarization
* text-generation
* translation
* zero-shot-classification

##### Zero-shot Classification
We'll start by tackling a more challenging task where we need to classify texts that haven't been labeled. This is a common scenario in real-world projects because annotating text is usually time-consuming and requires domain expertise. For this use case, the zero-shot-classification pipeline is very powerful: it allows you to specify which labels to use for the classification, so you don't have to rely on the labels of the pretrained model. You've already seen how the model can classify a sentence as positive or negative using those two labels - but it can also classify the text using any other set of labels you like.
```
from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define the input text and candidate labels
text = "The company reported a significant increase in revenue for the second quarter."
candidate_labels = ["finance", "sports", "technology"]

# Perform zero-shot classification
result = classifier(text, candidate_labels)
print(result)
```

Output:

```
{
    'sequence': 'The company reported a significant increase in revenue for the second quarter.',
    'labels': ['finance', 'technology', 'sports'],
    'scores': [0.993, 0.005, 0.002]
}
```

In this example, the model correctly identifies the text as related to "finance" with a high confidence score.
This pipeline is called zero-shot because you don't need to fine-tune the model on your data to use it. It can directly return probability scores for any list of labels you want!
##### Text Generation
Now let's see how to use a pipeline to generate some text. The main idea here is that you provide a prompt and the model will auto-complete it by generating the remaining text. This is similar to the predictive text feature that is found on many phones. Text generation involves randomness, so it's normal if you don't get the same results as shown below.
````
from transformers import pipeline

generator = pipeline("text-generation")
generator("Once upon a time in a magical forest,")
````

output: [{'generated_text': 'Once upon a time in a magical forest, a mysterious woman was found on the verge of being attacked. After a momentary thought, we can see the source of this mysterious female energy in her right hand. Her power was quite clear already, there'}]

You can control how many different sequences are generated with the argument num_return_sequences and the total length of the output text with the argument max_length.

```
from transformers import pipeline

generator = pipeline("text-generation", max_length = 15 , num_return_sequences = 2 )
generator("Once upon a time in a magical forest,")
```

output: [{'generated_text': 'Once upon a time in a magical forest, there is a small magical boy'},
 {'generated_text': 'Once upon a time in a magical forest, a man named Shinnob'}]
 
##### Using any model from the Hub in the pipeline

The previous examples used the default model for the task at hand, but you can also choose a particular model from the Hub to use in a pipeline for a specific task - say, text generation. Go to the Model Hub and click on the corresponding tag on the left to display only the supported models for that task. You should get to a page like this one.

```
from transformers import pipeline

# Load the BART model for text summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text to summarize
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize 
its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" 
functions that humans associate with the human mind, such as "learning" and "problem-solving".
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=25)
print(summary)
```

output: [{'summary_text': 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive"'}]

You can refine your search for a model by clicking on the language tags, and picking a model that will generate text in another language. The Model Hub even contains checkpoints for multilingual models that support several languages.

Once you select a model by clicking on it, you'll see that there is a widget enabling you to try it directly online. This way you can quickly test the model's capabilities before downloading it.

##### The Inference API
All the models can be tested directly through your browser using the Inference API, which is available on the Hugging Face website. You can play with the model directly on this page by inputting custom text and watching the model process the input data.
##### Question Answering
The question-answering pipeline answers questions using information from a given context:

```
from transformers import pipeline

# Load the DistilBERT model for question answering
question_answerer = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Example context and question
context = """
The Apollo program was the third United States human spaceflight program carried out by NASA, which accomplished landing the first humans on the Moon from 1969 to 1972.
First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft to follow the one-person Project Mercury, which put the first Americans in space,
Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in an address to Congress on May 25, 1961.
"""

question = "Who was president when the Apollo program was first conceived?"

# Generate answer
answer = question_answerer(question=question, context=context)
print(answer)
```

output:
{'score': 0.4981418550014496, 'start': 193, 'end': 213, 'answer': 'Dwight D. Eisenhower'}
Note that this pipeline works by extracting information from the provided context; it does not generate the answer.


##### Named Entity Recognition

Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let's look at an example:

```
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True , )
ner("Hugging Face Inc. is a company based in New York City. Its founders include Clément Delangue, Julien Chaumond, and Thomas Wolf.")
```


output:

```
[
  {'entity_group': 'ORG', 'score': 0.998, 'word': 'Hugging Face Inc.', 'start': 0, 'end': 17},
  {'entity_group': 'LOC', 'score': 0.999, 'word': 'New York City', 'start': 37, 'end': 50},
  {'entity_group': 'PER', 'score': 0.998, 'word': 'Clément Delangue', 'start': 73, 'end': 89},
  {'entity_group': 'PER', 'score': 0.999, 'word': 'Julien Chaumond', 'start': 91, 'end': 106},
  {'entity_group': 'PER', 'score': 0.999, 'word': 'Thomas Wolf', 'start': 112, 'end': 123}
]
```

When working with Named Entity Recognition (NER), entities detected by the model can sometimes be split into multiple tokens due to the nature of tokenization. For example, the name "Hugging Face Inc." might be split into separate tokens like "Hugging", "Face", and "Inc.". The grouped_entities=True parameter in the Hugging Face pipeline helps to group these sub-tokens back into coherent entities. Here's why this is important and how it works:
##### **Why use Grouped Entities?**
###### Coherent Entity Representation:
* **Original Sentence:** "Hugging Face Inc. is a company based in New York City."
* **Without Grouping:** The model might output separate tokens like Hugging, Face, and Inc. each classified separately.
* **With Grouping:** These tokens are combined back into Hugging Face Inc. as a single entity, making the output more meaningful and easier to understand.

**Conclusion**
In this chapter, we've explored the basics of NLP and introduced the powerful capabilities of Transformer models. We looked at practical examples of text classification, named entity recognition, Question Answering, and text summarization, showcasing how these models can be applied to solve real-world problems using the Hugging Face pipeline.

**What's Next?**
The pipelines we've used so far are highly effective for specific tasks but may not be flexible enough for variations or more complex needs. In the next chapter, we'll dive deeper into the pipeline() function to understand what's happening under the hood. You'll learn how to customize pipelines to fit your specific requirements and explore the core components that make these models so powerful.

Don't hesitate to experiment with other pipelines on your own. This will help you gain more hands-on experience and further deepen your understanding of Transformer models and their applications in NLP.

Stay tuned for the next installment of "NLP Exploration: Chapter by Chapter with Me," where we'll continue our journey into the fascinating world of Transformer models!

#NLP #MachineLearning #AI #Transformers #HuggingFace