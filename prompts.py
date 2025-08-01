prompt_decompose = """You are a helpful assistant for image editing. 
    I will provide you with a caption that describes an image,
    and an editing instruction that represents a subjective modification
    of the scene, i.e. something that impacts the scene in its entirety. 
    Your task is to propose objective modifications. 
    You can ask to add or replace elements in the scene,
    proposing consistent modification that agree with the
    global instruction. Be concise and output your instructions without
    further considerations or reasoning, one local modification per line.
    Do not output any other text than the suggested outputs, do not write
    "suggested output:".
    I am going to provide some examples now.

    Caption: a photo of a urban scenario, with cars. 
    Global instruction: make the scene vintage. 
    Suggested output: replace the cars with old cars

    Caption: a photo of a dog running on the grass. 
    Global instruction: make it look funny. 
    Suggested output: add a hat to the dog

    The main subject of the scene must stay the same. For instance, if the photo is describing a cat
    as the main subject, you cannot replace the cat with another animal. You should NEVER remove elements.
    Only propose instructions targeting elements that appear in the caption, without imagining anything else.

    Now, provide {} outputs for the following caption and subjective instruction.
    Caption: {} Subjective instruction {}"""

prompt_caption_gpt = "I am going to provide an input image and an editing instruction." \
                  "You should propose 1) a caption that describes accurately the input image, max 10 words, focusing only on visual content." \
                  "2) a caption that encompasses how the image should look like after applying the instruction. The instruction is: \"{}\"." \
                  "Try to keep these captions as compact as possible. The captions should be as similar as possible to each other." \
                  "You should reply following the format: 1. <caption 1> 2. <caption 2>" \
                  "Just reply with the captions without reasoning or considerations. [image]"
