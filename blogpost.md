# BLIP-2.3: Improving BLIP-2's OK-VQA performance with GPT-3's world knowledge and in-context learning capabilities
> Authors: J. Belleman, M.J.A. Bosch, D.G. van Dijk, E.R. Langezaal & T. Veenboer \
> Deep Learning 2 - University of Amsterdam
## Table of contents

 - [1 Introduction](#1-introduction)
 - [2 Background Literature](#2-background-literature)
 - [3 Related Work](#3-related-work)
   - [3.1 CLIP Descriptor Classification](#31-clip-descriptor-classification)
   - [3.2 Expert Ensembles](#32-expert-ensembles)
   - [3.3 PromptCap: Captioning Images Through Finetuning](#33-promptcap-captioning-images-through-finetuning)
 - [4 Strengths and Weaknesses of BLIP-2](#4-strengths-and-weaknesses-of-blip-2)
   - [4.1 Strengths](#41-strengths)
   - [4.2 Weaknesses](#42-weaknesses)
 - [5 Datasets](#5-datasets)
   - [5.1 Evaluation metric](#51-evaluation-metric)
 - [6 Reproduction](#6-reproduction)
   - [6.1 Results from original paper](#61-results-from-original-paper)
   - [6.2 Reproduced results](#62-reproduced-results)
   - [6.3 Reproduction difficulties](#63-reproduction-difficulties)
 - [7 Extending BLIP-2 with GPT-3](#7-extending-blip-2-with-gpt-3)
   - [7.1 Problem description](#71-problem-description)
   - [7.2 BLIP-2.3 Pipeline](#72-blip-23-pipeline)
     - [7.2.1 Hyperparameters](#721-hyperparameters)
 - [8 Results and Analysis](#8-results-and-analysis)
   - [8.1 Quantitative Results](#81-quantitative-results)
   - [8.2 Qualitative Results](#82-qualitative-results)
 - [9 Conclusion](#9-conclusion)
   - [9.1 Future Work](#91-future-work)
     - [9.1.1 Combining VQA and Captioning](#911-combining-vqa-and-captioning)
     - [9.1.2 Larger LLM's for Baseline VQA](#912-larger-llms-for-baseline-vqa)
 - [10 Ablation studies](#10-ablation-studies)
   - [10.1 Approach 1: Image specific VQA context](#101-approach-1-image-specific-vqa-context)
   - [10.2 Approach 2: Salient noun prompting](#102-approach-2-salient-noun-prompting)
 - [References](#references)

## 1 Introduction

The field of vision-language pre-training (VLP)[^2] has seen significant progress in recent years, with researchers employing pre-trained models of increasingly larger scale. However, to effectively leverage pre-trained unimodal models for vision-language tasks, it is essential to facilitate cross-modal alignment.

In this research we highlight the novel BLIP-2[^5] architecture; the proposed BLIP-2 method introduces a new two-stage pre-training strategy to achieve cross-model alignment between frozen unimodal models. This is significant, as it has the potential to accelerate progress in the development of vision-language models, enabling the deployment of such models in real-world applications such as visual question answering. BLIP-2 already outperforms the state-of-the-art on zero-shot visual question answering with 57x fewer trainable parameters[^5], allowing researchers with limited computing resources to contribute to developments in this area. 

For this project we focus on extending BLIP2. BLIP-2 represents a generic and efficient pre-training strategy that bootstraps vision-language pre-training from frozen pre-trained image encoders and frozen large language models. To bridge the modality gap it utilises a Querying Transformer (Q-Former) which is trained in a representation learning and generative learning stage.
We create BLIP-2.3 (the combination of BLIP-2 and GPT-3[^1]) with the aim to enhance the performance of the BLIP-2 model on vision-language tasks. This improvement will be achieved by leveraging the learning and language generation capabilities of GPT-3[^1] through its API. The goal is to unlock in-context learning capabilities in BLIP-2.3 in order to achieve better performance.

The overarching objective of this research is to harness the respective strengths of BLIP-2 and GPT-3 in order to enhance BLIP-2's visual question answering capabilities. By leveraging the open-world knowledge and in-context learning capabilities of GPT-3, we aim to improve BLIP-2's performance on the OK-VQA dataset[^11]. This study will try to provide insights into how the strengths of different models can be combined to achieve better performance on complex tasks.

## 2 Background Literature
We are focusing on the subdomain of artificial intelligence known as multi-modal learning. Specifically, we will focus on the combination of vision and language. Powerful neural vision networks have existed for a couple years, achieving breakthroughs with Convolution Neural Networks[^4] and more recently by employing Vision Transformers[^3]. Especially in language it has been shown that incredible performance can be achieved using ever larger models (LLM's) that can capture and comprehend more contextual information. Current state of the art is achieved by GPT-3[^1], a language model that can perform a plethora of tasks. Unlike smaller open source models like GPT-2[^25], the GPT-3 model features the emergent capability of in-context learning and is able to perform various tasks without expensive finetuning, which we would not be able to do on our limited hardware. Compared to other LLM's of similar size, like Flamingo[^27], GPT-4[^26] or PaLM[^28], GPT-3 does allow for accessible and cheap online inference through its API, eliminating the need for massive processing power to load such models.

A downside of this increase in complexity is that finetuning and training these models is becoming increasingly resource intensive, motivating research into using pretrained models and therefore the research field known as Vision Language Pretraining (VLP)[^2]. It has already been shown that this combination can learn to perform both vision-language understanding and generation tasks, for example with the BLIP architecture[^6]. 

## 3 Related Work

### 3.1 CLIP Descriptor Classification
Vondrick and Menon[^13] present a method to enhance zero-shot classification performance of CLIP[^14] on various datasets such as ImageNet and EuroSAT. Their alteration to CLIP consists of producing a set of different descriptors for a given category. For the classification of an arbitrary image, the embeddings of the image are compared with the embeddings of these descriptors. The class with the highest image-versus-descriptor similarity score is chosen as the most probable category for the image. Unfortunately, their work does not extend to the domain of open-ended question answering, as there is no prospect of generating descriptors for certain answers, because the set of possible answers a model can give is infinite.

### 3.2 Expert Ensembles
In a recent paper, Liu et al. [^15] showcase a model that leverages an ensemble of pre-trained expert models to create several vision-language capabilities. Their full model, Prismer, relies on the efficiency of other models that have been trained to be state-of-the-art in their respective modality. For example, the set of expert models includes networks that examine objects within an image, do Optical Character Recognition (OCR) or distinguish the various parts of an image (segmentation). The combined information from these models is then processed by a smaller sized network which translates the information into text encodings; this is similar to the Q-Former's functionality in BLIP-2. Akin to BLIP-2, the smaller model then proceeds to pass the encodings to a language decoder that outputs a prediction, caption or answer to a visual question. It might be interesting to explore the possibilities of adding such ensembles of experts to the vision stage of BLIP-2 along with the suggested adjustments in this paper.

### 3.3 PromptCap: Captioning Images Through Finetuning
Hu et al. [^16] demonstrated that it is possible to conveniently take advantage of the world knowledge of GPT-3 with their question-aware captioning model PromptCap. The PromptCap model is a small trainable model, which generates context for the question that GPT-3 answers at a later stage. For instance, a sample question for an image of a microscope could be: 'Who invented this apparatus?'. PromptCap then generates a small caption belonging to the image which could be along the lines of: 'This is a researcher looking through a microscope'. GPT-3 is then prompted with the combination of the question and caption. The question and caption are sufficient for GPT-3 to answer: 'A Dutch spectacle maker named Zacharias Janssen'. The PromptCap model is similar to BLIP-2 to a high degree. Instead of applying the power of a pre-trained Vision Model (VM), PromptCap transforms the image to text immediately by finetuning to the dataset. Our research aims to use GPT-3 in a similar manner. However rather than fine-tuning, we utilize GPT-3's in-context learning capabilities by answering questions through few-shot learning based on BLIP-2's image captions. 

## 4 Strengths and Weaknesses of BLIP-2

### 4.1 Strengths
BLIP-2 is designed with a modular framework; both the VM and LM are interchangeable frozen models. The future will likely see more lightweight or powerful models, which can be employed with BLIP-2 by simply re-training the Q-Former. To illustrate, if OpenAI decides to release GPT-3 at a later point in time, the Q-Former can easily be adjusted to function in a pipeline with GPT-3 as frozen LM. On top of that, the Q-Former itself is a relatively small transformer that can be trained without the resources that are required for present-day LLMs. This opens up areas of research for less affluent institutions or companies that are otherwise out-of-scope. 

### 4.2 Weaknesses
Nevertheless, BLIP-2 has a few significant drawbacks in terms of efficiency. Firstly, BLIP-2 currently does not benefit from in-context learning examples. The authors ascribe this to the fact that the model has been trained on a dataset which solely contains single image-text data pairs [^5]. They also address this by stating that they intend to train the model on a different, more expressive dataset in subsequent research. Moreover, BLIP-2 often has an inadequate response to questions that test its world knowledge. An example of such a question could be about the logo of a well-known company. BLIP-2 frequently does not recognize logos or brand marks because it is trained to dissect purely visual attributes of an image. Models large enough to have a vast source of world knowledge are usually closed source or too large to be deployed on a low-cost configuration. This research specifically fixates on intjecting GPT-3 into the pipeline of BLIP-2 to increase its world knowledge. Our intention is to enable GPT-3 to reason directly about the visual cues it receives from BLIP-2.

## 5 Datasets
As outlined in the introduction we focus specifically on Visual Question Answering[^7] datasets. These datasets consists of images, for example extracted from the COCO dataset[^8], each with various questions about the image and corresponding human-annotated ground truth answers. These answers are often only a few words, such that some form of quantitative evaluation can be performed by matching a model's output with these answers. 

The original BLIP-2 paper[^5] outlines 3 such datasets, which we also use for reproduction. Since the original VQA-dataset released by Antol et. al.[^9] contains some questions which can easily be answered without viewing the image[^10], the curated VQA-V2 dataset[^10] is used. A lot of the questions in this dataset however are simple classification tasks, focussing merely on attribute recognition, counting or object detection. Therefore, the OK-VQA dataset[^11] is also used, which focusses specifically on questions that require external knowledge. Therefore, the OK-VQA dataset is especially interesting to test whether the world knowledge stored in GPT-3 can be used to improve BLIP-2 on this VQA task, and is therefore the sole dataset used to test our extension. Following the same logic, the GQA dataset[^12] is also used in the original paper, which too focusses on real-world visual reasoning and compositional question answering as oppposed to simple object recognition.



| VQA-V2  | OK-VQA  | GQA |
| ------- | ------- | --- |
| ![](/images/vqav2-example.png)  | ![](/images/ok-vqa1.png)  |  ![](/images/gqa-example.png)  |

While these datasets are manually curated using tools such as Amazon Mechanical Turk[^20], the ground truth answers for the visual questions are from rather varying quality. Often, the human answers are factually incorrect, such as the answer "Africa" to the question "What south american country usually has this climate?". It is important to keep this in mind, as the vast amount of domain knowledge of GPT-3 can lead to better answers than given by the human annotators, which leads to underestimation of the world knowledge of the model. 

### 5.1 Evaluation metric

All models are evaluated using the same evaluation metric on these VQA datasets. The accuracy is determined by dividing the number of annotators out of ten who gave the same answer as the model by 3. Therefore, a full accuracy of 1 is given when at least three annotators gave the same answer and an accuracy of 66\% when two out of ten annotators agree on the given answer. Pre-processing such as lowercasing and removing punctuation is applied on the answers before exact matching between the ground truth answers and the model's answer. The final evaluation metric reported is simply the average accuracy over all datapoints in the test/validation set. One sidenote is that the OK-VQA dataset only had five annotators per question so they doubled each answer.

## 6 Reproduction

### 6.1 Results from original paper
The reproduction goal of our research focuses on a specific part of the results presented in the BLIP-2 paper. The original paper evaluates the performance of the BLIP-2 model on a variety of tasks such as visual question answering, image captioning and image-text retrieval. Since this research aims to enhance the in-context learning capabilities of BLIP-2 by combining its strenghts with those of the GPT-3 LLM, we are interested in the performance of models on the VQA tasks. The achieved VQA results on the datasets mentioned in the previous section are presented for various models in table 2 of the original paper, which is duplicated below.

<p align="center">
  <img src="./images/reproduction_table.png">
</p>

### 6.2 Reproduced results
The red boxes indicate the results that we attempted to reproduce in our work. We focus on the bottom section of the table since only the BLIP-2 model itself is within the scope of this research, and most models in the top section are closed source as well. For the frozen vision transformer, we only utilize the pretrained model with the ViT-G[^22] encoder for reproduction, since the BLIP-2 pretrained model with the ViT-L encoder from CLIP[^21] was not available to us. As for the frozen large language models, we include both the OPT[^23] as well as the FlanT5[^24] model in our reproduction study. For both models, only the smaller variants are tested, namely the 2.7B and XL versions for OPT and FlanT5 respectively, since our computational resources made it unfeasible to test their larger counterparts (6.7B and XXL variants).

Our reproduction results are presented in the table below. Based on the results in the table, we can state that the accuracies of the OPT and FlanT5 BLIP-2 model variants on the VQA tasks are succesfully reproducable.

<table align="center">
   <thead>
      <tr>
         <th>Models</th>
         <th>VQAv2 (val)</th>
         <th>OK-VQA</th>
         <th>GQA</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>BLIP-2 ViT-G OPT<sub>2.7B</sub></td>
         <td>53.4</td>
         <td>31.8</td>
         <td>34.6</td>
      </tr>
      <tr>
         <td>BLIP-2 ViT-G FlanT5<sub>XL</sub></td>
         <td>61.8</td>
         <td>39.3</td>
         <td>43.9</td>
      </tr>
   </tbody>
</table>
  
### 6.3 Reproduction difficulties
The LAVIS library by Salesforce [^17] provides an out-of-the-box approach to evaluating BLIP-2 with various different frozen language models. Through a single python file, evaluate.py, an end-user can easily configure which model they want to evaluate. The different LLM's all come with their respective configuration file. Moreover, the library supplies python scripts for downloading the different datasets that can be used with BLIP.  

Nevertheless, the reproduction of BLIP-2 posed a few issues. First of all, the authors mention in the paper that their prompt template differs for OPT and FlanT5. The prompt for OPT is supposedly 'Question: {}. Answer:', while FlanT5's prompt should be 'Question: {}. Short answer:'. However, we found in the LAVIS repository - maintained by the authors of BLIP2 - that the prompt is identical for both models, namely 'Question: {}. Short answer'. Furthermore, the method of retrieval for the FlanT5 model in the library ensures that it can only be used with the datatype bfloat16 (Brain Floating Point) [^18]. The bfloat16 datatype has been introduced by the Google Brain team to achieve higher performance with less memory requirements than a standard float32. Modern-day GPUs have the capability of performing matrix multiplications with bfloat16; however, older GPUs do not always possess this ability. The GPU provided by our cluster was not able to execute the evaluation script with bfloat16. Therefore, we had to add and register the FlanT5 model with 8-bit integer weights in a separate file. Fortunately, LAVIS supplies the user with the effortless extensibility of registering new models by adding a single line.

The reproduction of BLIP-2 with the OPT model presented another difficulty. Initially, the LAVIS library was incompatible with the latest transformers library [^19] (above version >= 4.27 at the time of writing). This led to the inaccessibility of several methods of the OPT model, which in turn caused the evaluation script to throw various exceptions. After communicating with the developers of LAVIS, this was resolved swiftly with a code update to the library.

## 7 Extending BLIP-2 with GPT-3

### 7.1 Problem description
The performance of BLIP-2 in visual question answering tasks is limited due to the inaccurate world knowledge of the large language model. As a result, despite correctly obtaining the visual information, BLIP-2 may generate incorrect inferences and ultimately produce unsatisfactory answers. For instance, BLIP-2 might be able to effectively recognize the object depicted in an image, but its reasoning process may fail to correctly answer a related question. An example of this can be seen in the figure below, where BLIP-2's line of reasoning falls short since it does not consider weather circumstances of the location mentioned.

<p align="center">
  <img src="./images/blip_reasoning_example.jpeg">
</p>

### 7.2 BLIP-2.3 Pipeline
In this research project we aim at tackling this main bottleneck of the BLIP-2 model by combining BLIP-2's advanced image captioning capabilities with the general real-world knowledge of GPT-3. The extension is focused on augmenting the performance of the model on the OK-VQA dataset, which is specifically designed to necessitate external knowledge to answer the posed questions. We will utillize the GPT-3 model by feeding it visual information extracted from BLIP-2 to generate answers to the OK-VQA instances.

We do so by leveraging the GPT-3 API to incorporate in-context learning in a distinct way. Namely, we allow BLIP-2 to generate a general image caption to provide GPT-3 with enough visual context to answer the OK-VQA question. Finally, we combine the image caption and the OK-VQA question and input them into the GPT-3 model. The GPT-3 model then leverages its comprehensive world knowledge and the extracted visual information to answer the OK-VQA question. To enable in-context learning, we first augment the GPT-3 model's capabilities by providing a few example inputs and outputs. It is important to note that the effectiveness of GPT-3 in answering the OK-VQA question depends on the available context. In situations where the provided context is inadequate for GPT-3 to generate an answer, GPT-3 indicates this limitation by stating it's uncertainty, and in such cases, we rely on the original answer generated by BLIP-2. 

#### 7.2.1 Hyperparameters
The temperature hyperparameter of the GPT-3 API determines the randomness in the output. Higher values give more chance to lower probability output and lower temperature makes GPT-3 more deterministic. The desired behaviour on OK-VQA, where GPT-3 has to respond with the most likely factual answer given the context, is that GPT-3 always outputs its most likely answer. This makes the predictions conservative and predictable such that if GPT-3’s most likely answer is unknown, then we can still use BLIP-2’s answer. In addition, setting the temperature to 0 leads to consistent answers across multiple runs, making our results reproducible. We also used a repetition penalty of 2 and length penalty of 1.5 for the generation of context by BLIP-2 as this gives medium to long sentence output without repetition of text. In our experiments, having BLIP-2 generate longer sentences often lead to inaccurate descriptions of the photo, where BLIP-2 would sometimes halliucinate. Finally, we used the same hyperparameter settings as the authors of the original paper when BLIP-2 generated the short baseline answers.

<p align="center">
  <img src="./images/BLIP23_pipeline.png">
</p>

## 8 Results and Analysis

### 8.1 Quantitative Results

The use of the description of a photo generated by the BLIP-2 FlanT5<sub>XL</sub> model, combined with a three-shot approach using GPT-3, has shown significant improvements in accuracy on the OK-VQA dataset. Our best method achieved a 5% increase in accuracy, reaching 44.37\% as shown in the table below. In comparison, the reproduction of BLIP-2 results achieved 39.3\% on the same test set. Unfortunately testing more than three-shot approaches was unfeasible, since the number of OpenAI API tokens used increases linearly with the number of in-context learning examples, making this increasingly costly.

As part of our ablation studies, we investigated the impact of moving from a zero-shot approach to a three-shot approach. In the three-shot approach, GPT-3 was provided with three examples from the training set and their corresponding ground truth answers. This resulted in an improved task performance from 42.91\% to 44.37\%. Furthermore, the FlanT5<sub>XL</sub> model fine-tuned on COCO captioning demonstrated worse performance compared to using the pre-trained FlanT5<sub>XL</sub> model (for both zero-shot and three-shot). We also experimented with having GPT-3 generate questions related to the original question, which BLIP-2 then answered to form a context for GPT-3 to summarize. This approach, detailed in the penultimate row, performed even worse than the baseline. Another approach, which also included BLIP-2’s description of the most salient noun of a question as context for GPT-3 resulted in decreased performance, with 40.62\% accuracy, compared to the methods that only relied on the photo description. However, it still achieved 1.3\% higher accuracy than the baseline. More details regarding the last two mentioned ablations and methods are provided in the [ablation studies](#10-ablation-studies) section. 


<table align="center">
   <thead>
      <tr>
         <th>Models</th>
         <th>Provided context of image?</th>
         <th>GPT-3: Few-shot or zero-shot?</th>
         <th>OK-VQA (test)</th>
      </tr>
   </thead>
   <tbody>
      <tr>
        <td>ViT-G FlanT5<sub>XL</sub></td>
         <td>Photo description</td>
         <td>Three-shot</td>
        <td><b>44.37</b></td>
      </tr>
       <tr>
         <td>ViT-G FlanT5<sub>XL</sub></td>
         <td>Photo description</td>
         <td>Zero-shot</td>
         <td>42.91</td>
      </tr>
          <tr>
         <td>ViT-G Flant5<sub>XL</sub> COCO finetuned</td>
         <td>Photo description</td>
         <td>Three-shot</td>
         <td>41.62</td>
      </tr>
      <tr>
         <td>ViT-G Flant5<sub>XL</sub></td>
         <td>description generated to GPT-3's questions</td>
         <td>Zero-shot</td>
         <td>38.72</td>
      </tr>
      <tr>
         <td>ViT-G FlanT5<sub>XL</sub></td>
         <td>Photo description + noun description</td>
         <td>Three-shot</td>
         <td>40.62</td>
      </tr>
   </tbody>
</table>

### 8.2 Qualitative Results

To gain a better understanding of the performance improvements achieved by our designed pipeline for BLIP-2, it is crucial to examine specific examples from the OK-VQA dataset. The following graph shows two examples from the OK-VQA dataset, highlighting the image and its corresponding question. The first set of orange text bars represent the answers generated by BLIP-2, both of which were found to be incorrect. Then the second set of orange text bars represent the image caption that is generated by BLIP-2. The final answers of GPT-3, generated through the use of this image caption and the OK-VQA question, is shown in purple. 

<p align="center">
  <img src="./images/qualitative_main_approach.png">
</p>

As shown in the illustration, both answers were succesfully improved by GPT-3. In the left image example, GPT-3 effectively combined the visual information of the elephant in the image with the question that inquired about the appropriate term for someone handling that specific animal. Leveraging its extensive world-knowledge, GPT-3 correctly identified that a handler of elephants is called a "mahout." On the other hand, BLIP-2 failed to provide the correct answer to this question, potentially due to its limited understanding of real-world concepts such as the definition of a "mahout." Moving to the right image example, GPT-3 once again demonstrated its capability to improve the answer. By utilizing its knowledge that sheep are commonly found in New Zealand rather than Australia, GPT-3 accurately deduced the location of the depicted sheep. In contrast, BLIP-2 was unable to provide an accurate response to this question.

These instances exemplify how GPT-3's integration into our pipeline has enhanced the model's performance by leveraging its broad range of knowledge and contextual understanding. The successful improvements exhibited by GPT-3 in these examples highlight the significance of incorporating a deep learning model with rich world-knowledge into the visual question answering task.

## 9 Conclusion

This study demonstrated that GPT-3's in-context learning capabilities can help to improve the performance of BLIP-2 on the OK-VQA dataset. This has been accomplished by prompting GPT-3 in an efficient few-shot manner whilst supplying it with limited information about an image. The introduction of GPT-3 into the BLIP-2 pipeline has shown that inaccuracies in terms of world knowledge in a smaller-sized LLM can be resolved by invoking a larger, more well-rounded model. Furthermore, we have shown that prompting GPT-3 zero-shot is worse than few-shot. This reinforces the fact that GPT-3 has access to in-context learning to improve upon itself. 

### 9.1 Future Work

#### 9.1.1 Combining VQA and Captioning

This study has shown that the utilization of straightforward captions surpasses all other methods of inference in the BLIP-2.3 pipeline. Currently, the captions are generated by the FlanT5 model finetuned for question answering. It might be interesting to substitute this model with the FlanT5 that has been finetuned on the task of captioning. Nevertheless, the FlanT5 model for question answering is essential for correcting GPT-3 in cases where it is uncertain. Therefore, the suggestion is to devise a pipeline where both models are incorporated to maximize GPT-3's potential. Due to the time constraint of our current course and a limitation of computational resources, we were not able to explore this combination to its full extent.

#### 9.1.2 Larger LLM's for Baseline VQA

Throughout this blogpost, the performed research revolved around the usage of FlanT5<sub>XL</sub>. In [reproduction](#6-reproduction) we highlighted the fact that the original BLIP-2 paper contained larger and more effective models for text generation. However, the access to computational resources at an academic institution is unfortunately limited; we were therefore unable to load more substantial models onto the available GPU's. We theorize that having a stronger baseline for question answering in the form of FlanT5<sub>XXL</sub> or OPT<sub>6.7</sub> would have a significant impact on performance of the BLIP-2.3 pipeline. Moreover, the first [approach](#101-approach-1-image-specific-vqa-context) in the ablation studies might be effective if the underlying question-answering model is more proficient at providing descriptive answers to GPT's questions. Lastly, the quality of captions could also possibly see extensive improvement by employing a larger LLM finetuned to captioning.

## 10 Ablation studies
In this section, we present a comprehensive analysis of two alternative methods that were employed and their corresponding outcomes and challenges encountered. The purpose of these ablation studies was to evaluate the effectiveness and limitations of different approaches in addressing the problem at hand. 

### 10.1 Approach 1: Image specific VQA context
We initially investigated the feasibility of utilizing GPT-3 to generate specific questions that are essential for providing a meaningful answer to the OK-VQA question. These generated questions were then passed to BLIP-2, leveraging its visual question answering capabilities, in order to obtain additional visual context for the GPT-3 model. The obtained answers from BLIP-2, along with their corresponding questions and the original VQA-question, were subsequently inputted into the GPT-3 model to generate the final answer. However, this approach exhibited numerous inaccuracies due to BLIP-2's inability to successfully address the highly specific questions generated by GPT-3. As a result, either limited or inaccurate visual information was provided, where BLIP-2 occasionally fabricated responses that lacked factual accuracy. This ultimately resulted in this ablation getting only 38.72\% accuracy on OK-VQA, which is lower than than the BLIP-2 baseline.

<p align="center">
  <img src="./images/pipeline_ablation1.png">
</p>


### 10.2 Approach 2: Salient noun prompting
Furthermore, we explored an alternative approach of letting GPT-3 pick the most salient noun within an OK-VQA question. To accomplish this, we presented GPT-3 with a set of example questions paired with their corresponding target nouns, leveraging the in-context learning capabilities of GPT-3. The selected noun was then employed to construct a more context-specific prompt for BLIP-2, enabling it to generate an image caption that specifically highlights the relevant portion of the image necessary for answering the OK-VQA question. This method, for which the pipeline is depicted in the following graph, exhibited an improvement in the performance of the BLIP-2 FlanT5<sub>XL</sub> model on the OK-VQA dataset, with accuracy rising from 39.3% to 40.6%. However, despite this improvement, there were still instances where the performance of the model remained suboptimal.

<p align="center">
  <img src="./images/pipeline_ablation2.png">
</p>

As mentioned, salient noun prompting did, on average, improve the accuracy on the OK-VQA dataset but overall did not outperform our simpler approach. To understand this, we need to look at  a few example results from this approach.  The following graph again shows two examples from the OK-VQA dataset, highlighting the image and its corresponding question. The first set of orange text bars represent the answers generated by BLIP-2, whilst the first set of purple text bars represents the salient noun that GPT-3 selected. This is followed by the second set of orange text bars, containing the noun-specific context that BLIP-2 generated. The final answers of GPT-3 is represented by the last set of purple text bars.

<p align="center">
  <img src="./images/qualitative_noun_approach.png">
</p>


As shown in the graph, the left OK-VQA example was indeed improved by noun specific prompting, whilst the right example did in fact get worse. In the left image example, the noun selected by GPT-3 helps BLIP-2 look in the relevant region of the image, thereby providing a good context, which GPT-3 can use to succesfully answer the question. However, in the right image example, BLIP-2 mentions that this person would be a lazy person, thereby, through incorrect context, directly leading GPT-3 towards a wrong answer to the question. These kind of examples show that BLIP-2 might sometimes confuse GPT-3 by providing too much nonfactual context, which results in a lower overall accuracy score.

Overall, it was shown that the simpler approach yielded the best performance, primarily due to BLIP-2's limited ability to generate accurate and truthful context when presented with highly specific prompts or questions. The misleading and inaccurate contextual information provided by BLIP-2 had a detrimental effect on GPT-3, leading to poorer performance for both of the explored approaches.

## References
[^1]: Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Amodei, D. (2020). Language models are few-shot learners.

[^2]: Chen, F.-L., Zhang, D.-Z., Han, M.-L., Chen, X.-Y., Shi, J., Xu, S., Xu, B.
(2023). Vlp: A survey on vision-language pre-training. *Machine Intelligence Research*, 20 (1), 38–56.

[^3]: Han, K., Wang, Y., Chen, H., Chen, X., Guo, J., Liu, Z. (2022). A survey on vision transformer. *IEEE transactions on pattern analysis and
machine intelligence*, 45 (1), 87–110

[^4]: Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In F. Pereira,
C. Burges, L. Bottou, K. Weinberger (Eds.), *Advances in neural information processing systems* (Vol. 25). Curran Associates, Inc.

[^5]: Li, J., Li, D., Savarese, S., Hoi, S. (2023). Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.

[^6]: Li, J., Li, D., Xiong, C., Hoi, S. (2022). Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. *
International conference on machine learning* (pp. 12888–12900).

[^7]: Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., & Parikh, D. (2015). VQA: Visual question answering. *Proceedings of the IEEE international conference on computer vision* (pp. 2425-2433).

[^8]: Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., & Zitnick, C. L. (2014). Microsoft coco: Common objects in context. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13* (pp. 740-755). Springer International Publishing.

[^9]: Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., & Parikh, D. (2015). Vqa: Visual question answering. In *Proceedings of the IEEE international conference on computer vision* (pp. 2425-2433).

[^10]: Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In *CVPR*, pp. 6325–6334, 2017

[^11]: Marino, K., Rastegari, M., Farhadi, A., & Mottaghi, R. (2019). Ok-vqa: A visual question answering benchmark requiring external knowledge. In *Proceedings of the IEEE/cvf conference on computer vision and pattern recognition* (pp. 3195-3204).

[^12]: Hudson, D. A., & Manning, C. D. (2019). Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 6700-6709).

[^13]: Menon, S., & Vondrick, C. (2022). Visual Classification via Description from Large Language Models. ArXiv, abs/2210.07183.

[^14]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PMLR.

[^15]: Liu, S., Fan, L., Johns, E., Yu, Z., Xiao, C., & Anandkumar, A. (2023). Prismer: A Vision-Language Model with An Ensemble of Experts. ArXiv Preprint ArXiv:2303. 02506.

[^16]: Hu, Y., Hua, H., Yang, Z., Shi, W., Smith, N. A., & Luo, J. (2022). PromptCap: Prompt-Guided Task-Aware Image Captioning. ArXiv Preprint ArXiv:2211. 09699.

[^17]: Li, D., Li, J., Le, H., Wang, G., Savarese, S., & Hoi, S. C. H. (2022). LAVIS: A Library for Language-Vision Intelligence.

[^18]: N. Burgess, J. Milanovic, N. Stephens, K. Monachopoulos and D. Mansell, "Bfloat16 Processing for Neural Networks," 2019 IEEE 26th Symposium on Computer Arithmetic (ARITH), Kyoto, Japan, 2019, pp. 88-91, doi: 10.1109/ARITH.2019.00022.

[^19]: Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., … Rush, A. M. (2020, October). Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45.

[^20]: Crowston, K. (2012). Amazon Mechanical Turk: A Research Tool for Organizations and Information Systems Scholars. In: Bhattacherjee, A., Fitzgerald, B. (eds) Shaping the Future of ICT Research. Methods and Approaches. IFIP Advances in Information and Communication Technology, vol 389. Springer, Berlin, Heidelberg.

[^21]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021). Learning transferable visual models from natural language supervision.

[^22]: Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., and Cao, Y. Eva. (2022). Exploring the limits of masked visual representation learning at scale. arXiv preprint arXiv:2211.07636.

[^23]: Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M. T., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. (2022). OPT: open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

[^24]: Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., Zhao, V. Y., Huang, Y., Dai, A. M., Yu, H., Petrov, S., Chi, E. H., Dean, J., Devlin, J., Roberts, A., Zhou, D., Le, Q. V., and Wei, J. (2022). Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

[^25]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

[^26]: OpenAI. (2023). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774

[^27]: Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., … Simonyan, K. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. arXiv preprint arXiv:2204.14198

[^28]: Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., … Fiedel, N. (2022). PaLM: Scaling Language Modeling with Pathways. arXiv preprint arXiv:2204.02311
