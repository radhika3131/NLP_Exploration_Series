# Chapter 2: How do Transformers work?
Welcome back to "NLP Exploration: Chapter by Chapter with Me!" In Chapter 1, we introduced the basics of NLP and explored some practical applications of Transformer models. Now, it's time to understand how these powerful models work under the hood. In this chapter, we will take a high-level look at the architecture of Transformer models.
## A Brief History of Transformers
Transformers have a fascinating history that highlights their rapid evolution and impact on the field of NLP. Here are some key milestones:
**June 2017**: The Transformer model was introduced in the paper "Attention is All You Need" by Vaswani et al. The focus of the original research was on translation tasks and the results were remarkable. This was followed by the introduction of several influential models, including:
* **June 2018**: GPT (Generative Pre-trained Transformer), the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results
* **October 2018**: BERT (Bidirectional Encoder Representations from Transformers), another large pretrained model, this one designed to produce better summaries of sentences (more on this in the next chapter!)
* **February 2019**: GPT-2, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns
* **October 2019**: DistilBERT, a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT's performance
* **October 2019**: BART and T5, two large pretrained models using the same architecture as the original Transformer model (the first to do so)
* **May 2020**, GPT-3, an even bigger version of GPT-2 that can perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)

This timeline highlights some of the major developments in Transformer models. Broadly, Transformer models can be grouped into three categories:
* **GPT-like (Auto-regressive Transformer models)**
* **BERT-like (Auto-encoding Transformer models)**
* **BART/T5-like (Sequence-to-sequence Transformer models)**

We will dive into these families in more depth later on.
## Transformers are Language Models
All the Transformer models we've mentioned, like GPT, BERT, BART, and T5, are types of language models. This means they've been trained on huge amounts of text data in a way that doesn't need humans to label the data manually. This type of training is called self-supervised learning.
### What is Self-Supervised Learning?
In self-supervised learning, the model learns by predicting parts of the text based on other parts. For example, if the model sees a sentence, it might try to guess the next word in the sentence. This way, the model can learn patterns and structures in the language all on its own.
### Why Self-Supervised Learning?
The big advantage here is that we don't need humans to manually label the data, which can be time-consuming and expensive. The model teaches itself from the text it reads, developing a statistical understanding of the language.
### From General to Specific Tasks: Transfer Learning
After training on general text, these models become good at understanding language, but they might not be very useful for specific tasks right away. To make them useful for tasks like translation, summarization, or question answering, they go through another process called transfer learning.
In transfer learning, the model is fine-tuned with labeled data for a specific task. For instance, if we want the model to summarize texts, we show it examples of long texts paired with their summaries. The model learns from these examples and gets better at the task.
#### Example: Predicting the Next Word
One simple task these models can do is predict the next word in a sentence. For example, if the model has read the sentence "The cat sat on the," it might predict the next word to be "mat." This is called causal language modeling because the model uses the words it has seen so far to predict the next word, the output depends on the past and present inputs, but not the future ones.
Another example is masked language modeling. In this approach, some words in a sentence are hidden (masked), and the model's job is to predict the missing words. For instance, if the sentence is "The cat sat on the [MASK]," the model should predict that the masked word is "mat." This technique helps the model understand the context and relationships between words in a sentence.
## Transformers are Big Models
### Increasing Model Size for Better Performance
One of the primary strategies to enhance the performance of Transformer models is to increase their size. This includes making the models larger regarding the number of parameters they contain and the amount of data they are trained on. Larger models capture more complex patterns and nuances in the data, which leads to better performance on various tasks.
### Why Bigger is Often Better
When we talk about increasing the size of Transformer models, we're usually referring to the number of layers, the size of each layer, and the number of parameters. More parameters mean the model has more capacity to learn from data, allowing it to understand and generate more accurate predictions.
### The Cost of Training Large Models
However, training large models comes with significant costs:
* **Data Requirements**: Training a large model requires a vast amount of data. Gathering, cleaning, and preparing this data can be a massive undertaking.
* **Computational Resources**: Larger models need more computational power. Training these models involves running complex calculations on high-performance hardware, such as GPUs (Graphics Processing Units) or TPUs (Tensor Processing Units). This process can take days, weeks, or even months, depending on the model size and available resources.
* **Time**: The training process for large models is time-consuming. It involves multiple iterations over the data to adjust the model parameters and optimize performance.
* **Environmental Impact**: The significant computational resources required translate to high energy consumption. This has an environmental impact, as the energy used often comes from non-renewable sources. Training a single large model can result in a considerable carbon footprint.

We will dive into the carbon footprint of transformers in more depth later on.
## Understanding Transfer Learning: Pretraining and Fine-Tuning
**Pretraining** is the initial phase of training a model from scratch. Here's how it works:
* **Random Initialization**: The model's weights are randomly initialized at the start.
* **Large Datasets**: The model is trained on vast amounts of data, which can encompass general language data or domain-specific data.
* **Long Training Duration**: This phase requires extensive computational resources and can take several weeks or even months.

The objective of pretraining is to create a model that has a broad understanding of language by learning patterns, structures, and nuances from the extensive dataset.
**Fine-tuning**, on the other hand, is the training done after a model has been pretrained. Here's the process:
* **Acquiring a Pretrained Model**: Start with a model that has already undergone the extensive pretraining phase.
* **Task-Specific Data**: Use a smaller, task-specific dataset to further train the model. This dataset is much smaller compared to the dataset used in pretraining.
Shorter Training Duration: Fine-tuning is significantly faster and requires fewer resources than pertaining.

### Why Not Train from Scratch?
Training a model from scratch for a specific task might seem straightforward, but there are compelling reasons to use transfer learning:
* **Leverage Pretrained Knowledge**: The pretrained model has already learned a general understanding of the language, which can be transferred to the specific task. For example, a model pretrained on general English text can be fine-tuned on a specific scientific corpus like arXiv papers to perform well in scientific text analysis.
* **Reduced Data Requirements**: Fine-tuning requires much less data. Since the model already understands general language patterns, it needs fewer examples to adapt to the specific task.
* **Cost and Resource Efficiency**: Pretraining is resource-intensive, both in terms of time and computational power. Fine-tuning, on the other hand, is faster and cheaper.
* **Better Performance**: Models fine-tuned on specific tasks tend to perform better than those trained from scratch unless you have an enormous amount of data for the specific task.

### Example of Transfer Learning
Consider a scenario where you need a model to understand and generate scientific research papers. Instead of training a model from scratch:
* **Pretraining**: Use a large dataset of general English text to pre-train a language model like BERT or GPT.
* **Fine-Tuning**: Fine-tune the pretrained model using a smaller dataset of arXiv research papers. This will help the model adapt to the specific language and style used in scientific research.

By doing this, the model will quickly and efficiently learn to handle scientific texts with high accuracy, leveraging the general language understanding it gained during pretraining.
Transfer learning, through pretraining and fine-tuning, is a powerful technique in machine learning that enables efficient and effective model training. By leveraging pretrained models, you can achieve better performance on specific tasks with reduced data, time, and resources. This makes it a preferred approach in various NLP applications.
### **General Architecture of Transformer**
In this section, we'll go over the general architecture of the Transformer model. Don't worry if you don't understand some of the concepts; In the next chapter, we will go through detailed sections covering each of the components.
![](https://github.com/radhika3131/NLP_Exploration_Series/blob/main/images/Ar2Img1.png)
The model is primarily composed of two blocks:
* Encoder (left): The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
* Decoder (right): The decoder uses the encoder's representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

Each of these parts can be used independently, depending on the task:
* Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
* Decoder-only models: Good for generative tasks such as text generation.
* Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

We will dive into those architectures independently in the next chapter.

## Attention layers
A key feature of Transformer models is that they are built with special layers called attention layers. The title of the paper introducing the Transformer architecture was "Attention Is All You Need"! We will explore the details of attention layers later in the course; for now, all you need to know is that this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.

To put this into context, consider the task of translating text from English to French. Given the input "You like this course", a translation model will need to also attend to the adjacent word "You" to get the proper translation for the word "like", because in French the verb "like" is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating "this" the model will also need to pay attention to the word "course", because "this" translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of "course". With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.

The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.

Now that you have an idea of what attention layers are all about, let's take a closer look at the Transformer architecture.

### The Orginal Architecture
The Transformer model is like a smart translator. It has two main parts: an encoder and a decoder. Let's see how they work together to translate a sentence.
**Encoder**
* **Purpose**: The encoder reads the entire sentence in the original language (e.g., English) and understands its meaning.
* **How it works**: It uses "attention layers" to look at all the words in the sentence at the same time. This way, it can figure out how each word relates to the others.

**Decoder**
* **Purpose**: The decoder takes the encoder's understanding of the sentence and translates it into the target language (e.g., French).
* **How it works**: The decoder works one word at a time, using the words it has already translated to guess the next word. For example, if it has translated "I am" into "Je suis," it will use these words to figure out what comes next.

**Training the Model**

When training the model, both the encoder and decoder see the complete sentences. However, the decoder can only use the words it has already translated. This prevents it from cheating by looking ahead at future words.

**Example:**

If the decoder is predicting the fourth word, it can only look at the first three words it has translated.

The original Transformer architecture looked like this, with the encoder on the left and the decoder on the right:
![](https://github.com/radhika3131/NLP_Exploration_Series/blob/main/images/Art2Img2.png)

Note that the first attention layer in a decoder block pays attention to all (past) inputs to the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. This is very useful as different languages can have grammatical rules that put the words in different orders, or some context provided later in the sentence may be helpful to determine the best translation of a given word.

### Attention Masks

To help the model focus on the right parts of the sentence, we use attention masks:
* **Padding Words**: When translating multiple sentences at once, some sentences might be shorter. We add padding words to make them the same length and use masks to ignore these padding words.
* **Sequential Attention**: In the decoder, masks ensure it only looks at words it has already translated, keeping the sequence correct.

In simple terms, the Transformer model uses the encoder to understand the entire input sentence and the decoder to translate it one word at a time. Attention layers help it focus on important parts of the sentence, and masks ensure it processes words in the right order without looking ahead. This method is powerful because it can handle complex sentence structures and relationships between words, making it effective for translation tasks.

In the next chapter, we'll dive deeper into the specifics of the encoder, decoder, and sequence-to-sequence models. We'll break down how each component functions and how they come together to perform complex language tasks.

Stay tuned for more insights and practical techniques as we continue our journey through the fascinating world of Transformers.
#NLP #Transformers #MachineLearning #DeepLearning #AI #HuggingFace #SelfAttention #LanguageModels #Pretraining #FineTuning #SequenceToSequence