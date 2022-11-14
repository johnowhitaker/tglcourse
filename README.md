The Generative Landscape
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This is still a work in progress, but most of the important bits should
be done in time for a launch on Friday, Novermer 25! In the meantime,
you can check out my [previous course,
AIAIART](https://github.com/johnowhitaker/aiaiart) or join the Discord
where I’ll be streaming final lesson checks and updates. <br>

If you want to support this effort, I now have a Patreon:
<https://www.patreon.com/johnowhitaker> <br>

The material will show in github pages at
<https://johnowhitaker.github.io/tglcourse/> (and the
http://thegenerativelandscape.com will redirect there now too). Hooray
for the magic of nbdev. <br>

Join our [Discord](https://discord.gg/vSjhr8xb4g) to discuss the course,
join study groups or chat about all things generative. That’s also the
place to go for notifications of live lesson walkthroughs and course
updates.

![](index_files/figure-gfm/cell-2-output-1.png)

Check out the [Getting Started](./00_Getting_Started.ipynb) page for an
overview of the course and more information on things like study
groups.<br>

Check out the
[Library](https://johnowhitaker.github.io/tglcourse/library.html) page
for information on the `tglcourse` library that accompanies the course.

### The Plan

The idea is to have a core curriculum building up to an understanding of
key generative modelling techniques, split into three rough sections.
The first 5 lessons will cover basics of building NNs and crafting loss
functions. Lessons 6-10 will introduce generative modelling, GANs and
working with sequences using transformers. Lessons 11-15 will be a deep
dive into diffusion models, and lesson 16 will wrap up and give
sugestions for new directions to explore.

Alongside this will be a number of ‘bonus’ notebooks that don’t need to
be completed but which augment the core content. Managing datasets,
experiment tracking, sharing demos and so on. Some will augment specific
lessons, some will add functionality to the library and some will just
be standalone topics I think are cool. The latter category is likely to
continue to grow even after the course launches :)

There will be three suggested projects (after lessons 5, 11 and 14) to
mark milestones in the course, and the final lesson will also encourage
you to do a larger project at the end. Once we launch there’ll be a
place to everyone’s projects and some prizes for the best.

This table has a rough status on the main lessons.

| Lesson                              | Description                                                                                                                                      | Status           |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| Lesson 1: PyTorch Basics            | Intro to PT, tensor manipulation, images as tensors, LeastAverageImage exercise                                                                  | Done             |
| Lesson 2: Optimization              | Intro to GD, optimization examples exercise                                                                                                      | Rough Draft Done |
| Lesson 3: Building NNs              | nn.Module, building blocks, CNNs                                                                                                                 | Rough Draft Done |
| Lesson 4: Learning Representations  | What do networks learn, style transfer                                                                                                           | WIP              |
| Lesson 5: CLIP                      | Demo use as a loss function, video deep dive into VQGAN notebooks                                                                                | Rough Draft Done |
| Lesson 6: Generative Modelling      | VAE part, latent walks, PCA                                                                                                                      | WIP              |
| Lesson 7: GANs 1                    | Intro to GANs, DC GAN, Conditioning                                                                                                              | WIP              |
| Lesson 8: GANs 2                    | GAN training tricks, NOGAN, using modern GANs, VQGAN                                                                                             | Not Started      |
| Lesson 9: Sequence Modelling Intro  | idea, language modelling concept, transformer demo                                                                                               | WIP              |
| Lesson 10: Transformers             | Intro to transformes, attention, comparing to lstm, reading minGPT                                                                               | Not Started      |
| Lesson 11: Everythign is a sequence | Show whistlegen, protein, VQGAN, parti…                                                                                                          | Not Started      |
| Lesson 12: DM 1                     | Intro to diffusion models, toy example, comparison to DDPM                                                                                       | Rough Draft Done |
| Lesson 13: DM2                      | Conditioning, CFG, guiding, sampling, better training                                                                                            | WIP              |
| Lesson 14: DM3                      | SD deep dive                                                                                                                                     | Rough Draft Done |
| Lesson 15: DM4                      | Other modalities - ideally demo class-conditioned audio generation. Might not be done by the time the course launches but would be nice to have. | WIP              |
| Lesson 16: Going Further            | Finding your niche, exploring less common areas                                                                                                  | WIP              |

### FAQs

Some course-related questions that have tricked in: <br> + ‘Any
prerequisites?’: If you’re comfortable with a bit of Python and using
Jupyter Notebooks you should be ready to take this course. No prior deep
learning knowledge is assumed, and although we will dive fairly deep
fairly quickly I’ve tried to link lots of resources wherever possible.
<br> + ‘How long is the course?’: There are 15 core lessons plus a
number of bonus notebooks. You can take them at whatever pace you find
enjoyable, or join in with a study group on Discord to work through one
a week.<br> + ‘Does the HuggingFace Diffusion Model Class supercede
this?’: I’ve teamed up with HF to help build their [diffusion model
class](https://github.com/huggingface/diffusion-models-class), sharing a
lot of material between it and this course. Both will have unique things
to add - I’d recommend signing up for theirs even if you’re working
through ‘The Generative Landscape’ as well, since there will be extra
projects and fun community activities to get involved in with that one
too. <br> + ‘Can genereative landscape be real?’: Yes, if you subscribe
to [David Chalmers ‘simulation
realism’](https://www.thephilosopher1923.org/post/taking-simulation-seriously)
;)<br> + ‘Will it include stable diffusion?’: Yes, see lesson 14<br> +
‘Will we learn how to adapt these method to 3d?’: At some point I’d love
to add more 3D related content - stay tuned for bonus notebooks once the
craziness of the course launch calms down.<br> + ‘I hope this includes
generating text!’: It does! Lessons 9 - 11 deal with modelling
sequences, and should set you up with everything you need to make models
which can spew out AI-generated gibberish all day long.<br> +‘My
question isn’t in the FAQs?’: OK, so I made this one up. But if you have
a burning question, send it to me and I’ll add it here.<br>

CI status:
[![](https://github.com/johnowhitaker/tglcourse/actions/workflows/test.yaml/badge.svg)](https://github.com/johnowhitaker/tglcourse/actions/workflows/test.yaml)

Page stats: Total Hits:
[![HitCount](https://hits.dwyl.com/johnowhitaker/tglcourse.svg?style=flat-square&show=unique)](http://hits.dwyl.com/johnowhitaker/tglcourse)
Page visitors: ![visitor
badge](https://page-views.glitch.me/badge?page_id=tglcourse.index)
