# BLIP-2.3: Improving BLIP-2's OK-VQA performance with GPT-3's world knowledge and in-context learning capabilities
> Authors: J. Belleman, M.J.A. Bosch, D.G. van Dijk, E.R. Langezaal & T. Veenboer \
> Deep Learning 2 - University of Amsterdam
## Table of contents

 * [Introduction](#introduction)
  * [Background Literature](#background-literature)
  * [Related Work](#related-work)
    + [CLIP Descriptor Classification](#clip-descriptor-classification)
    + [Expert Ensembles](#expert-ensembles)
    + [PromptCap: Captioning Images Through Finetuning](#promptcap-captioning-images-through-finetuning)
  * [Strengths and Weaknesses of BLIP-2](#strengths-and-weaknesses-of-blip-2)
    + [Strengths](#strengths)
    + [Weaknesses](#weaknesses)
  * [Datasets](#datasets)
  * [Reproduction](#reproduction)
    + [Results from original paper](#results-from-original-paper)
    + [Reproduced results](#reproduced-results)
    + [Reproduction difficulties](#reproduction-difficulties)
  * [Extending BLIP-2 with GPT-3](#extending-blip-2-with-gpt-3)
    + [Problem description](#problem-description)
    + [BLIP 2.3 Pipeline](#blip-23-pipeline)
  * [Results and Analysis](#results-and-analysis)
    + [Quantitative Results](#quantitative-results)
    + [Qualitative Results](#qualitative-results)
  * [Conclusion](#conclusion)
    + [Future Work](#future-work)
      - [Combining VQA and Captioning](#combining-vqa-and-captioning)
      - [Larger LLM's for Baseline VQA](#larger-llms-for-baseline-vqa)
  * [Ablation studies](#ablation-studies)
    + [Approach 1: Image specific VQA context](#approach-1-image-specific-vqa-context)
    + [Approach 2: Salient noun prompting](#approach-2-salient-noun-prompting)
  * [References](#references)

## Introduction

The field of vision-language pre-training (VLP)[^2] has seen significant progress in recent years, with researchers employing pre-trained models of increasingly larger scale. However, to effectively leverage pre-trained unimodal models for vision-language tasks, it is essential to facilitate cross-modal alignment.

In this research we highlight the novel BLIP-2[^5] architecture; the proposed BLIP-2 method introduces a new two-stage pre-training strategy to achieve cross-model alignment between frozen unimodal models. This is significant, as it has the potential to accelerate progress in the development of vision-language models, enabling the deployment of such models in real-world applications such as visual question answering. BLIP-2 already outperforms the state-of-the-art on zero-shot visual question answering with 57x fewer trainable parameters[^5], allowing researchers with limited computing resources to contribute to developments in this area. 

For this project we focus on extending BLIP2. BLIP-2 represents a generic and efficient pre-training strategy that bootstraps vision-language pre-training from frozen pre-trained image encoders and frozen large language models. To bridge the modality gap it utilises a Querying Transformer (Q-Former) which is trained in a representation learning and generative learning stage.
We create BLIP-2.3 (the combination of BLIP-2 and GPT-3) with the aim to enhance the performance of the BLIP-2 model on vision-language tasks. This improvement will be achieved by leveraging the learning and language generation capabilities of GPT-3 through its API. The goal is to unlock in-context learning capabilities in BLIP-2.3 in order to achieve better performance.

Overall, this research aims to leverage the strengths of both BLIP-2 and GPT-3 to enhance the in-context learning capabilities of BLIP-2 and improve its performance on (zero-shot) Visual Question Answering (VQA[^7]) tasks. This study will try to provide insights into how the strengths of different models can be combined to achieve better performance on complex tasks.

## Background Literature
We are focusing on the subdomain of artificial intelligence known as multi-modal learning. Specifically, we will focus on the combination of vision and language. Powerful neural vision networks have existed for a couple years, achieving breakthroughs with Convolution Neural Networks[^4] and more recently by employing Vision Transformers[^3]. Especially in language it has been shown that incredible performance can be achieved using ever larger models (LLM's) that can capture and comprehend more contextual information. Current state of the art is achieved by GPT-3[^1], a language model that can perform a plethora of tasks. A downside of this increase in complexity is that finetuning and training these models is becoming increasingly resource intensive, motivating research into using pretrained models and therefore the research field known as Vision Language Pretraining (VLP)[^2]. It has already been shown that this combination can learn to perform both vision-language understanding and generation tasks, for example with the BLIP architecture[^6]. 

## Related Work

### CLIP Descriptor Classification
Vondrick and Menon[^13] present a method to enhance zero-shot classification performance of the CLIP[^14] on various datasets such as ImageNet and EuroSAT. Their alteration to CLIP consists of producing a set of different descriptors for a given category. For the classification of an arbitrary image, the embeddings of the image are compared with the embeddings of these descriptors. The class with the highest image-versus-descriptor similarity score is chosen as the most probable category for the image. Unfortunately, their work does not extend to the domain of open-ended question answering. There is no prospect of generating descriptors for certain answers because the set of possible answers a model can give is infinite.

### Expert Ensembles
In a recent paper, Liu et al. [^15] showcase a model that leverages an ensemble of pre-trained expert models to create several vision-language capabilities. Their full model, Prismer, relies on the efficiency of other models that have been trained to be state-of-the-art in their respective modality. For example, the set of expert models can include a variety of networks that examine objects within an image, that can do Optical Character Recognition (OCR) and are capable of distinguishing the various parts of an image (segmentation). The combined information these models are then processed by a smaller sized network which translates the information into text encodings; this is alike the Q-Former's functionality in BLIP-2. Akin to BLIP, the smaller model then proceeds to pass the encodings to a language decoder that outputs a prediction, caption or answers a visual question. It might be interesting to explore the possibilities of adding such ensembles of experts to the vision stage of BLIP along with the suggested adjustments in this paper.

### PromptCap: Captioning Images Through Finetuning
Hu et al. [^16] demonstrated that the world knowledge of GPT-3 can be conveniently taken advantage of with their question-aware captioning model PromptCap. The PromptCap model is a small trainable model which generates context for the question GPT-3 answers at a later stage. For instance, a sample question for an image of a microscope could be: 'Who invented this apparatus?'. PromptCap then generates a small caption belonging to the image which could be along the lines of: 'This is a researcher looking through a microscope'. Thereafter, GPT-3 is prompted with the combination of the question and caption. The question and caption are sufficient for GPT-3 to answer: 'A Dutch spectacle maker named Zacharias Janssen'. The PromptCap model is similar to BLIP-2 to a high degree. Instead of applying the power of a pre-trained Vision Model (VM), PromptCap transforms the image to text immediately by finetuning to the dataset. Our research builds on the usage of GPT-3 in related fashion. However rather than fine-tuning, we utilize GPT-3's in-context learning capabilities by answering questions with few-shot learning based on BLIP-2's image captions. 

## Strengths and Weaknesses of BLIP-2

### Strengths
BLIP-2 is designed with a modular framework; both the VM and LM are interchangeable frozen models. The future will likely see more lightweight or powerful models, which can be employed with BLIP-2 by simply re-training the Q-Former. To illustrate, if OpenAI decides to release GPT-3 at a later point in time, the Q-Former can easily be adjusted to function in a pipeline with GPT-3 as frozen LM. On top of that, the Q-Former itself is a relatively small neural network that can be trained without the resources that are required for present-day LLMs. This opens up areas of research for less affluent institutions or companies that are otherwise out-of-scope. 

### Weaknesses
Nevertheless, BLIP-2 has a few significant drawbacks in terms of efficiency. Firstly, BLIP-2 currently does not support in-context learning with the standard setup. The authors ascribe this to the fact that the model has been trained on a dataset which solely contains single image-text data pairs [^5]. They also address this by stating that they intend to train the model on a different, more expressive dataset in subsequent research. Moreover, BLIP-2 often has an inadequate response to questions that test the respondent's world knowledge. An example of such a question could be about the logo of a well-known company. BLIP-2 frequently does not recognize logos or brand marks because it is trained to dissect purely visual attributes of an image. Models that do have a vast source of world knowledge are usually closed source or too large to be deployed on a low-cost configuration. The research of this blogpost specifically fixates on interjecting GPT-3 into the workflow of BLIP-2 to increase the world knowledge of the model in its entirety. We intend to do so by directly letting GPT-3 reason about the visual cues it receives from the BLIP-2 pipeline.

## Datasets
As outlined in the introduction we focus specifically on Visual Question Answering[^7] datasets. These datasets consists of images, for example extracted from the COCO dataset[^8], each with various questions about an image and a bunch of exemplar answers per question. These answers are often only a few words, such that some form of quantitative evaluation can be performed by matching a model's output with these answers. 

The original BLIP-2 paper[^5] outlines 3 such datasets, which we also use for reproduction and our extension. Since the original VQA-dataset released by Antol et. al.[^9] contains some questions which can easily be answered without viewing the image[^10], the curated VQAv2 dataset[^10] is used. A lot of the questions in this dataset however are simple classification tasks, focussing merely on attribute recognition, counting or object detection. Therefore the OK-VQA dataset[^11] is also used, which focussed specifically on questions that require external knowledge. As such the OK-VQA dataset is especially interesting to test if we can harnest the plethora of real-world knowledge stored in GPT-3 with our extension. Following the same logic, the GQA dataset[^12] is also used, which too focusses on real-world visual reasoning and compositional question answering as oppposed to simple object recognition.



| VQA-V2  | OK-VQA  | GQA |
| ------- | ------- | --- |
| ![](/images/vqav2-example.png)  | ![](/images/ok-vqa1.png)  |  ![](/images/gqa-example.png)  |

While these datasets are manually curated using tools such as Amazon Mechanical Turk[^20], the label answers for the visual questions are from rather varying quality. In quite some cases some of the set of template answers for a question are blatantly wrong, such as answering "Africa" to the question "What south american country usually has this climate?". It is important to keep this mind, as the vast domain knowledge of GPT-3 can cause it give more precise/correct answers than the human annotaters, which are incorrectly counted as wrong because of this flawed gold standard. 

## Reproduction

### Results from original paper
The reproduction goal of our research focuses on a specific part of the results presented in the BLIP-2 paper. The original paper evaluates the performance of the BLIP-2 model on a variety of tasks such as visual question answering, image captioning and image-text retrieval. Since we aim in this research to enhance the in-context learning capabilities of BLIP-2 by combining its' strenghts with those of the GPT-3 LLM we are interested in the performance of models on the VQA task. The achieved results on this task of various models on the datasets mentioned in the previous section are presented in table 2 of the original paper and can be seen below.

<p align="center">
  <img src="./images/reproduction_table.png">
</p>

### Reproduced results
The red boxes indicate the results that we attempted to reproduce in our work. We focus on the bottom section of the table since only the BLIP-2 model itself is within the scope of this research. For the frozen vision transformer, we only utilize the pretrained model with the ViT-G[^22] encoder for reproduction, since the BLIP-2 pretrained model with the ViT-L encoder from CLIP[^21] was not available to us. As for the frozen large language models, we include both the OPT[^23] as well as the FlanT5[^24] model in our reproduction study. We test for both models only their smaller variants, namely the 2.7B and XL versions for OPT and FlanT5 respectively, since our computational resources limit us to not use their larger counterparts (6.7B and XXL variants).

Our reproduction results are presented in the table below. Based on the results in the table, we can state that the accuracies of the OPT and FlanT5 BLIP-2 model variants on the VQA task are succesfully reproducable.

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
  
### Reproduction difficulties
The LAVIS library by Salesforce [^17] provides an out-of-the-box approach to evaluating BLIP-2 with various different frozen Language Models. Through a single python file, evaluate.py, an end-user can easily configure which model that he wants to evaluate. The variety of LMs all come with their respective configuration file. Moreover, it supplies python scripts for downloading the different datasets that can be used with BLIP.  

Nevertheless, the reproduction of BLIP-2 posed a few issues. First of all, the authors mention in the paper that their prompt template differs for OPT and FlanT5. The prompt for OPT is supposedly 'Question: {}. Answer:', while FlanT5's prompt should be 'Question: {}. Short answer:'. However, we found in the LAVIS repository - maintained by the authors of BLIP2 - that the prompt is identical for both models, namely 'Question: {}. Short answer'. Furthermore, the method of retrieval for the FlanT5 model in the library ensures that it can only be used with the datatype bfloat16 (Brain Floating Point) [^18]. The bfloat16 dtype has been introduced by the Google Brain team to achieve higher performance with less memory requirements than a standard float32. Modern-day GPUs have the capability of performing matrix multiplications with the bfloat16; however, older GPUs do not always possess this ability. The GPU provided by our cluster was not able to execute the evaluation script with bfloat16. Therefore, we had to add and register the FlanT5 model with 8-bit integer weights in a separate file. Fortunately, LAVIS supplies the user with the effortless extensibility of registering new models by adding a single line.

The reproduction of BLIP-2 with the OPT model presented another difficulty. Initially, the LAVIS library was incompatible with the latest transformers library [^19] (above version >= 4.27 at the time of writing). This lead to the inaccessibility of several methods of the OPT model which in turn caused the evaluation script to throw various exceptions. After communicating with the developers of LAVIS, this was resolved swiftly with a code update to the library.

## Extending BLIP-2 with GPT-3

### Problem description
The performance of BLIP-2 in visual question answering is subject to limitations due to inaccurate knowledge from the Large Language Model. As a result, despite correctly obtaining the visual information, BLIP-2 may generate incorrect inferences and ultimately produce unsatisfactory answers. For instance, BLIP-2 might be able to effectively recognize the object depicted in an image, but its' reasoning process may fail to correctly answer a related question. An example of this can be seen in the figure below, where BLIP-2's line of reasoning falls short since it does not consider weather circumstances of the location mentioned.

<p align="center">
  <img src="./images/blip_reasoning_example.jpeg">
</p>

### BLIP 2.3 Pipeline
In this research project we aim at tackling this main bottleneck of the BLIP-2 model by combining BLIP-2's advanced visual question answering and image captioning capabilities with the general real-world knowledge of GPT-3.  The extension is primarily focused on augmenting the performance of the model on the OK-VQA dataset, which is specifically designed to necessitate external knowledge to answer the posed questions. We will utillize the GPT-3 model by feeding it visual information extracted from BLIP-2 to generate answers to the OK-VQA instances.

We do so by leveraging the GPT-3 API to incorporate in-context learning in a distinct way. Namely, we allow BLIP-2 to generate a general image caption to provide GPT-3 with enough visual context to answer the OK-VQA question. Finally, we combine the image caption and the OK-VQA question and input them into the GPT-3 model. The GPT-3 model then leverages its comprehensive world knowledge and the extracted visual information to answer the OK-VQA question. To enable in-context learning, we first augment the GPT-3 model's capabilities by providing a few example inputs and outputs. It is important to note that the effectiveness of GPT-3 in answering the OK-VQA question depends on the available context. In situations where the provided context is inadequate for GPT-3 to generate an answer, GPT-3 indicates this limitation, and in such cases, we rely on the original answer generated by BLIP-2.

## Results and Analysis

### Quantitative Results

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
        <td>ViT<sub>g</sub> FlanT5<sub>XL</sub></td>
         <td>Photo description</td>
         <td>three-shot</td>
        <td><b>44.37</b></td>
      </tr>
       <tr>
         <td>ViT<sub>g</sub> FlanT5<sub>XL</sub></td>
         <td>Photo description</td>
         <td>zero-shot</td>
         <td>42.91</td>
      </tr>
          <tr>
         <td>ViT<sub>g</sub> Flant5<sub>XL</sub> COCO finetuned</td>
         <td>Photo description</td>
         <td>three-shot</td>
         <td>41.62</td>
      </tr>
      <tr>
         <td>ViT<sub>g</sub> FlanT5<sub>XL</sub></td>
         <td>Photo description + noun description</td>
         <td>three-shot</td>
         <td>40.62</td>
      </tr>
       <tr>
         <td>ViT<sub>g</sub> Flant5<sub>XL</sub></td>
         <td>Answers to clarifying questions</td>
         <td>zero-shot</td>
         <td>x</td>
      </tr>
   </tbody>
</table>

### Qualitative Results

To gain a better understanding of the performance improvements achieved by our designed pipeline for BLIP-2, it is crucial to examine specific examples from the OK-VQA dataset. The following graph shows two examples from the OK-VQA dataset, highlighting the image and its corresponding question. The first set of orange text bars represent the answers generated by BLIP-2, both of which were found to be incorrect. Then the second set of orange text bars represent the image caption that is generated by BLIP-2. The final answers of GPT-3, generated trough the use of this image caption and the OK-VQA question, is shown in purple. 

<p align="center">
  <img src="./images/qualitative_main_approach.png">
</p>

As shown in the graph, both answers were succesfully improved by GPT-3. In the left image example, GPT-3 effectively combined the visual information of the elephant in the image with the question that inquired about the appropriate term for someone handling that specific animal. Leveraging its extensive world-knowledge, GPT-3 correctly identified that a handler of elephants is called a "mahout." On the other hand, BLIP-2 failed to provide the correct answer to this question, potentially due to its limited understanding of real-world concepts such as the definition of a "mahout." Moving to the right image example, GPT-3 once again demonstrated its capability to improve the answer. By utilizing its knowledge that sheep are commonly found in New Zealand rather than Australia, GPT-3 accurately deduced the location of the depicted sheep. In contrast, BLIP-2 was unable to provide an accurate response to this question.

These instances exemplify how GPT-3's integration into our pipeline has enhanced the model's performance by leveraging its broad range of knowledge and contextual understanding. The successful improvements exhibited by GPT-3 in these examples highlight the significance of incorporating a deep learning model with rich world-knowledge into the visual question answering task.

## Conclusion

### Future Work

#### Combining VQA and Captioning

This study has shown that the utilization of straightforward captions surpasses all other methods of inference in the BLIP-2.3 pipeline. Currently, the captions are generated by the FlanT5 model finetuned for questioning. It might be interesting to substitute this model with the FlanT5 that has been finetuned on the task of captioning. Nevertheless, the FlanT5 model for question answering is essential for correcting GPT-3 in cases where it is uncertain. Therefore, the suggestion is to devise a pipeline where both models are incorporated to maximize GPT-3's potential. Due to the time constraint of our current course and a limitation of computational resources, we were not able to explore this combination to its full extent.

#### Larger LLM's for Baseline VQA

Throughout this blogpost, the performed research revolved around the usage of FlanT5<sub>XL</sub>. In [Reproduction](#reproduction) we highlighted the fact that the original BLIP-2 paper contained larger and more effective models for both vision encoding and text generation. However, the access to computational resources at an academic institution is unfortunately limited; we were therefore unable to load more substantial models onto the available GPU's. We theorize that having a stronger baseline for question answering in the form of FlanT5<sub>XXL</sub> or OPT<sub>6.7</sub> would have a significant impact on performance of the BLIP-2.3 pipeline. Moreover, the first [approach](#approach-1-image-specific-vqa-context) in the ablation studies might be effective if the underlying question-answering model is more proficient at providing descriptive answers to GPT's questions. Lastly, the quality of captions could also possibly see extensive improvement by employing a larger LLM finetuned to captioning.

## Ablation studies
In this section, we present a comprehensive analysis of two alternative methods that were employed and their corresponding outcomes and challenges encountered. The purpose of these ablation studies was to evaluate the effectiveness and limitations of different approaches in addressing the problem at hand. 

### Approach 1: Image specific VQA context
We initially investigated the feasibility of utilizing GPT-3 to generate specific questions that are essential for providing a meaningful answer to the OK-VQA question. These questions, generated by GPT-3, were then passed to BLIP-2, leveraging its visual question answering capabilities, in order to obtain additional visual context for the GPT-3 model. The obtained answers from BLIP-2, along with their corresponding questions and the original VQA-question, were subsequently inputted into the GPT-3 model to generate the final answer. However, this approach exhibited numerous inaccuracies due to BLIP-2's inability to successfully address the highly specific questions generated by GPT-3. As a result, either limited or inaccurate visual information was provided, where BLIP-2 occasionally fabricated responses that lacked factual accuracy.
<p align="center">
  <img src="./images/pipeline_ablation1.png">
</p>


### Approach 2: Salient noun prompting
Furthermore, we explored an alternative approach of letting GPT-3 pick the most salient noun within an OK-VQA question.  To accomplish this, we presented GPT-3 with a set of example questions paired with their corresponding target nouns, leveraging the in-context learning capabilities of GPT-3. The selected noun was then employed to construct a more context-specific prompt for BLIP-2, enabling it to generate an image caption that specifically highlights the relevant portion of the image necessary for answering the OK-VQA question. This method, for which the pipeline is depicted in the following graph, exhibited an improvement in the performance of the BLIP-2 FlanT5<sub>XL</sub> model on the OK-VQA dataset, with accuracy rising from 39.3% to 40.6%. However, despite this improvement, there were still instances where the performance of the model remained suboptimal.

<p align="center">
  <img src="./images/pipeline_ablation2.png">
</p>

As mentioned, salient noun prompting did, on average, improve the accuracy on the OK-VQA dataset but overall did not outperform our more simpler approach. To understand this, we need to look at  a few example results from this approach.  The following graph again shows two examples from the OK-VQA dataset, highlighting the image and its corresponding question. The first set of orange text bars represent the answers generated by BLIP-2, whilst the first set of pruple text bars represents the salient noun that GPT-3 selected. This is followed by the second set of orange text bars, containing the noun-specific context that BLIP-2 generated. The final answers of GPT-3 is represented by the last set of purple text bars.

<p align="center">
  <img src="./images/qualitative_noun_approach.png">
</p>


As shown in the graph, the left OK-VQA example was indeed improved by noun specific prompting, whilst the right OK-VQA example did in fact worsen. In the left image example, the GPT-3 selected noun prompt helps BLIP-2 look in the relevant region of the image, thereby providing a good context to which GPT-3 can succesfully answer the question. However, in the right image example, BLIP-2 mentions that this person would be a lazy person, thereby, trough incorrect context, directly leading GPT-3 towards a wrong answer to the question. These kind of examples show that BLIP-2 might sometimes confuse GPT-3 by providing too much incorrect context, which results in a lower overall accuracy score.

Overall, it was determined that the simpler approach yielded the best performance, primarily due to BLIP-2's limited ability to generate accurate and truthful context when presented with highly specific prompts or questions. The misleading and inaccurate contextual information provided by BLIP-2 had a detrimental effect on GPT-3, leading to poor performance for both of the explored approaches.

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
