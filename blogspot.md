# BLIP-2.3: Improving BLIP-2 by employing ChatGPT3 in context learning
The field of vision-language pre-training (VLP)[^2] has seen significant progress in recent years, with researchers employing pre-trained models of increasingly larger scale. However, to effectively leverage pre-trained unimodal models for vision-language tasks, it is essential to facilitate cross-modal alignment.

In this research we highlight the novel BLIP-2[^5] architecture; the proposed BLIP-2 method introduces a new two-stage pre-training strategy to achieve cross-model alignment between frozen unimodal models. This is significant, as it has the potential to accelerate progress in the development of vision-language models, enabling the deployment of such models in real-world applications such as visual question answering. BLIP-2 already outperforms the state-of-the-art on zero-shot visual question answering with 57x fewer trainable parameters[^5], allowing researchers with limited computing resources to contribute to developments in this area. 

For this project we focus on extending BLIP2. BLIP-2 represents a generic and efficient pre-training strategy that bootstraps vision-language pre-training from frozen pre-trained image encoders and frozen large language models. To bridge the modality gap it utilises a Querying Transformer (Q-Former) which is trained in a representation learning and generative learning stage.
We create BLIP-2.3 (the combination of BLIP-2 and GPT-3) with the aim to enhance the performance of the BLIP-2 model on vision-language tasks. This improvement will be achieved by leveraging the learning and language generation capabilities of GPT-3 through its API. The goal is to unlock in-context learning capabilities in BLIP-2.3 in order to achieve better performance.

Overall, this research aims to leverage the strengths of both BLIP-2 and GPT-3 to enhance the in-context learning capabilities of BLIP-2 and improve its performance on (zero-shot) Visual Question Answering (VQA[^7]) tasks. This study will try to provide insights into how the strengths of different models can be combined to achieve better performance on complex tasks.

<br>

## Background Literature
We are focusing on the subdomain of artificial intelligence known as multi-modal learning. Specifically, we will focus on the combination of vision and language. Powerful neural vision networks have existed for a couple years, achieving breakthroughs with Convolution Neural Networks[^4] and more recently by employing Vision Transformers[^3]. Especially in language it has been shown that incredible performance can be achieved using ever larger models (LLM's) that can capture and comprehend more contextual information. Current state of the art is achieved by GPT3[^1], a language model that can perform a plethora of tasks. A downside of this increase in complexity is that finetuning and training these models is becoming increasingly resource intensive, motivating research into using pretrained models and therefore the research field known as Vision Language Pretraining (VLP)[^2]. It has already been shown that this combination can learn to perform both vision-language understanding and generation tasks, for example with the BLIP architecture[^6]. 

## Related Work

Vondrick and Menon[^13] present a method to enhance zero-shot classification performance of the CLIP[^14] on various datasets such as ImageNet and EuroSAT. Their alteration to CLIP consists of producing a set of different descriptors for a given category. For the classification of an arbitrary image, the embeddings of the image are compared with the embeddings of these descriptors. The class with the highest image-versus-descriptor similarity score is chosen as the most probable category for the image. Unfortunately, their work does not extend to the domain of open-ended question answering. There is no prospect of generating descriptors for certain answers because the set of possible answers a model can give is infinite.     

## Datasets
As outlined in the introduction we focus specifically on Visual Question Answering[^7] datasets. These datasets consists of images, for example extracted from the COCO dataset[^8], each with various questions and a bunch of exemplar answers per question. These answers are often only a few words, such that some form of quantitative evaluation can be performed by matching a model's output with these answers. 

The original BLIP-2 paper[^5] outlines 3 such datasets, which we also use for reproduction and our extension. Since the original VQA-dataset released by Antol et. al.[^9] contains some questions which can easily be answered without viewing the image[^10], the curated VQAv2 dataset[^10] is used. A lot of the questions in this dataset however are simple classification tasks, focussing merely on attribute recognition, counting or object detection. Therefore the OK-VQA dataset[^11] is also used, which focussed specifically on questions that require external knowledge. As such the OK-VQA dataset is especially interesting to test if we can harnest the plethora of real-world knowledge stored in GPT-3 with our extension. Following the same logic, the GQA dataset[^12] is also used, which too focusses on real-world visual reasoning and compositional question answering as oppposed to simple object recognition.



| VQA-V2  | OK-VQA  | GQA |
| ------- | ------- | --- |
| ![](/images/vqav2-example.png)  | ![](/images/ok-vqa1.png)  |  ![](/images/gqa-example.png)  |



## Reproduction
The reproduction goal of our research focuses on a specific part of the results presented in the BLIP-2 paper. The original paper evaluates the performance of the BLIP-2 model on a variety of tasks such as visual question answering, image captioning and image-text retrieval. Since we aim in this research to enhance the in-context learning capabilities of BLIP-2 by combining its' strenghts with those of the GPT3 LLM we are interested in the performance of models on the VQA task. The achieved results on this task of various models on the datasets mentioned in the previous section are presented in table 2 of the original paper and can be seen below.
![](/images/reproduction_table.png)

The red boxes indicate the results that we attempted to reproduce in our work. We focus on the bottom section of the table since only the BLIP-2 model itself is within the scope of this research. For the frozen vision transformer, we only utilize the ViT-g (ref) model for reproduction since the ViT-L (ref) model was not available to us. As for the frozen large language models, we include both the OPT (ref) as well as the FlanT5 (ref) model in our reproduction study. We test for both models only their smaller variants, namely the 2.7B and XL versions for OPT and FlanT5 respectively, since our compute resources limit us to not use their larger counterparts (6.7B and XXL variants).

## Pipeline (maybe rename?)

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
