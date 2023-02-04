# Image Naming Project

This is train and inference pipeline for the task of generation of multiple candidate captions for an image.  

**Note by 2023:** This is my first ML project \(created in 2020\), so it may contain poor quality/outdated code.  
Project idea taken from this [paper](https://arxiv.org/abs/1502.03044) and *Reference implementations* \(given below\).

## Dataset

The dataset included about 9.3 million *"image + one caption"* pairs from the [Conceptual Captions](https://aclanthology.org/P18-1238.pdf) and *IStock*. Validation set consisted of 10k samples. Each image was resized to 299\*299 resolution to speed up loading between experiments. Dataset must be converted to the format of the example files in the `./data` folder by using `utils.generate_json_data()`.  

After analyzing the captions, it turned out that ~15% contain words that need to be replaced or deleted - this is, for example, an indication of specific locations, names of something, dates, etc. To solve the problem of getting rid of all this baggage, the [Spacy](https://spacy.io/) was used, which replaces all the above cases with special tags. With it, we replaced the special tag *PERSON* with synonyms of this word \(for the sake of variety\), the special tag *CARDINAL* (denoting numbers) was simply removed. If the caption included any of the other 16 tags, then we did not take such a caption \(and therefore the image corresponding to it\) into the dataset, the same with too short captions \(the length is less than 4 tokens\).  

**Note by 2023:** Now there are much larger datasets, such as [LAION-5B](https://arxiv.org/pdf/2210.08402.pdf), I recommend to use it. Also today this architecture is very outdated relative to modern ones.

## Train

The training time for one epoch was ~50 hours on the *NVIDIA GeForce GTX 1080 Ti*.  
After 7 epochs validation **BLEU=0.308** was reached, [model weights](https://drive.google.com/file/d/1pO7rPpEPtPGCOPYCbpEk64hTLFvr2_ir/view?usp=share_link).  

## Beam Search Modification

To improve the quality and diversity of the captions candidates for single image, the ideas from this [paper](https://arxiv.org/pdf/1610.02424.pdf) were used, which boil down to automatically lowering the score of those words that have already been generated earlier. To regulate the length of the captions, a similar approach was used - to lower the score of the end-token in the first n steps, where n is the desired minimum length of the output captions. Testing has shown that the above operations should be carried out only for one of the internal candidates of the algorithm. Ultimately, with the help of the algorithm modifications described above, it was possible to achieve a much better quality of generated captions for the same model.

## Additional human evaluation

200 random test pictures with three candidate captions obtained from a model with different beam sizes were shown to nine experts. The experts rated each caption on a 5-point scale, in the same way the experts evaluated the ground truth captions. Next, the estimates were averaged \(**mean model captions estimate = 2.83**, **mean ground truth captions estimate = 4.13**)\, as a result, it was concluded that the captions of the model are of satisfactory quality, but still lag behind the quality of captions created by a human.

## Model conversion

There's a long way to get production-ready model in *tf-serving*:
1. [pytorch to onnx](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
2. [onnx to tf-protobuff](https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb)
3. [tf-protobuff to saved_model](https://medium.com/styria-data-science-tech-blog/running-pytorch-models-in-production-fa09bebca622)
4. [saved_model to tf-serving](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client_grpc.py)

Due to lack of LSTMCell layer support in [ONNX operations](https://onnx.ai/onnx/operators/index.html) I replaced it with LSTM layer, which led to minor changes in the code that do not affect performance.  
During the conversion, the models have extra inputs and outputs, which leads to an error when receiving a predict in serving. To fix this \(as well as to optimize the model itself\), you should use this [tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md). The following transformations were applied to the models: *strip_unused_nodes, remove_nodes, fold_constants, fold_batch_norms, fold_old_batch_norms, merge_duplicate_nodes*. After them, all models successfully reached *tf-serving*.

## Summary of improvements

What changes have I added to the code compared to previous implementations:
* Training on a much larger 9.3 mln dataset
* Beam search algorithm update for generation of the several different caption candidates
* Changing the dataset format to be able to work effectively with millions of samples
* Adding pipeline for model conversion for *tensorflow-serving*
* Replacing LSTMCell layer to LSTM for possibility ONNX model conversion
* Replacing image encoder to much stronger *resnext101_32x8d_wsl*

## Reference implementations

* [Pytorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
* [Show, Attend and Tell](https://github.com/AaronCCWong/Show-Attend-and-Tell)
* [Image Captioning Project](https://github.com/tkolanka/ece285_mlip_projectA)
