1. Rerunning experiments
This is ran on these software versions:
Python 3.10.12
torch 2.1.0+cu118
torchvision 0.16.0+cu118
matplotlib 3.7.1
numpy 1.23.5

To run the scale invariability test, select a style image and content image of different resolutions.
Run
$ python test.py --content YOUR_CONTENT_PATH --style YOUR_STYLE_PATH --output OUTPUT_PATH --style_img_weights 1.0

For example: !python test.py --content_dir input/content/ --style input/style/landscape.jpg --output out

You can observe in the output path that the style transfer is functional with different resolutions.

To run the content leak test, you can run this script to repeatedly style transfer the same object.
for i in range(20):
    !python test.py --content YOUR_IMAGE_PATH --style YOUR_STYLE_PATH --output YOUR_IMAGE_PATH
    time.sleep(60)
    #rename the output to the input
    !mv YOUR_IMAGE_PATH_NEW_FILENAME YOUR_IMAGE_PATH
    time.sleep(60)

To run the content aware position encoding test, take an image with repetitions and run the same style transfer algorithm.
Run 
$ python test.py --content YOUR_CONTENT_PATH --style YOUR_STYLE_PATH --output OUTPUT_PATH --style_img_weights 1.0

To run the style transfer algorithm with multiple styles, run
$ python test.py --content YOUR_CONTENT_PATH --style path_to_style_image1,path_to_style_image2,path_to_style_image3,etc... --output PATH --style_img_weights float1 float2 float3 etc..

Have the style images separated by commas with no spaces
The style image weights are float values separated by spaces

2. 
  a) The codebase was branched off from the StyTR-2 repository supplied by the paper StyTR2: Image Style Transfer with Transformers. https://github.com/diyiiyiii/StyTR-2
  b) The model files in ./models transformer.py and StyTR.py were modified. The StyTrans class was rewritten by me in the init section and the forward method to allow for multiple styles
and to take in weights for each style image. The forward method also has the style loss calculation rewritten as the calculations must be different because of multiple style images.
In the transformer.py file, the forward method was modified to allow for multiple styles and style image weights. This means the encoding and decoding methods were changed as well to allow for multiple styles and
its weights. Misc.py was also modified to accomodate the newer versions of torch, python and torchvision. This meant that many of the helper functions had to have portions rewritten because many 
torchvision and torch methods have been deprecated.
  c) The execution file test.py has been rewritten to run the style transfer algorithm with multiple style images and image weights.
3. Datasets: Download the content(COCO2014) and style(https://paperswithcode.com/dataset/wikiart) datasets, and put them into the folder "./datasets".

Download trained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  
[vit_embedding](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing), 
[decoder](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing), [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing) 
I've included them already and put them into the folder  ./experiments/

4. Video link to watch presentation:
https://www.youtube.com/watch?v=ToGPwcGGbcQ 
