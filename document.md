# From zero to something
This is not actual technical documentation, instead just a brief story of the challenges and things I learned while completing this project.

## Finding a model
Initially, I was gonna buy an API key for GPT4o because of its DALL-E model. I thought that it was the best out there for generating realistic looking images. I was wrong, every single image it generated had that "shiny" style with the blurred background. I wanted to get rid of it so I told GPT to unblur the background and it gave me this:
![ChatGPT's attempt at a photorealistic image](photos/unblur_attempt.png)

It turned out that the AI looking style is something that couldn't be changed if I stuck with DALL-E. Luckily, GPT suggested some other models and that's when I came across **Stable Diffusion XL** (SDXL). It was free (as long as you host it yourself), and could be fine-tuned to the photorealistic style. All I had to do was install it.

## Installing SDXL
This took way too long for no reason. I needed to install `automatic1111` which allows for easy image generation, and customization without having to write a line of code. I cloned the [repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and ran `pip install -r stable-diffusion-webui/requirements.txt"` to get all the dependencies. It was here that I ran into an error with two packages, pip could not build the wheel for tokenizers and pillow-avif-plugin. This had me stumped for so long, I tried every fix I could think of and that ChatGPT suggested. Turns out, all I had to do was create a venv (use Python3.10). After that, I once again installed the dependencies and dropped the [SDXL base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) I got off hugging face into `stable-diffusion-webui/models/Stable-diffusion`, and finally could get to generating an image.

## The first images
With the prompt: "A photo-realistic portrait of an 18 year old Pakistani male. He weighs 140lbs, height of 180cm, is slightly lean, palish-tan complexion, has dark brown curly hair that reaches eyebrows, is clean shaven, has a slightly large nose and slightly close together eyes, and has slightly larger lips." I clicked generate and saw my GPU usage jump to 100% and my VRAM max out. The result:

![The first image](photos/first_photo.png)

Was not what I was expecting. Trying the generation again with the same prompt I got: 

![The second image](photos/first_photo1.png)

Not really sure why it looks like a magazine cover.

At this point I thought that it might be over, luckily I remembered that ChatGPT suggested I use a pre-trained model called RealisticVision alongside SDXL which specializes in generating unrealistic images. I downloaded [version 5.1 Hyper](https://civitai.com/models/4201?modelVersionId=501240) and dropped it into `stable-diffusion-webui/models/Stable-diffusion`. 

## The second images
With the same prompt as before, these were some of my results:

![Image generated with Realistic Vision](photos/second_photo.png)

![Image generated with Realistic Vision](photos/second_photo1.png)

After tweaking some settings suggested [here](https://civitai.com/models/4201?modelVersionId=501240), I was able to get this:

![The best image yet](photos/second_photo2.png)

This actually looks real!


