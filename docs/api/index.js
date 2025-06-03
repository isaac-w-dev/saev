URLS=[
"saev/index.html",
"saev/nn/index.html",
"saev/nn/test_objectives.html",
"saev/nn/objectives.html",
"saev/nn/test_modeling.html",
"saev/nn/modeling.html",
"saev/app/index.html",
"saev/app/data.html",
"saev/app/modeling.html",
"saev/test_visuals.html",
"saev/helpers.html",
"saev/imaging.html",
"saev/interactive/index.html",
"saev/interactive/metrics.html",
"saev/interactive/features.html",
"saev/visuals.html",
"saev/test_config.html",
"saev/activations.html",
"saev/config.html",
"saev/colors.html",
"saev/training.html",
"saev/test_training.html",
"saev/test_activations.html"
];
INDEX=[
{
"ref":"saev",
"url":0,
"doc":"saev is a Python package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch. The main entrypoint to the package is in  __main__ ; use  python -m saev  help to see the options and documentation for the script.  Tutorials  Guide to Training SAEs on Vision Models 1. Record ViT activations and save them to disk. 2. Train SAEs on the activations. 3. Visualize the learned features from the trained SAEs. 4. (your job) Propose trends and patterns in the visualized features. 5. (your job, supported by code) Construct datasets to test your hypothesized trends. 6. Confirm/reject hypotheses using  probing package.  saev helps with steps 1, 2 and 3.  note  saev assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with  CUDA_VISIBLE_DEVICES=X to run on GPU X.  Record ViT Activations to Disk To save activations to disk, we need to specify: 1. Which model we would like to use 2. Which layers we would like to save. 3. Where on disk and how we would like to save activations. 4. Which images we want to save activations for. The  saev.activations module does all of this for us. Run  uv run python -m saev activations  help to see all the configuration. In practice, you might run:   uv run python -m saev activations \\  vit-family clip \\  vit-ckpt ViT-B-32/openai \\  d-vit 768 \\  n-patches-per-img 49 \\  vit-layers -2 \\  dump-to /local/scratch/$USER/cache/saev \\  n-patches-per-shard 2_4000_000 \\ data:imagenet-dataset   This will save activations for the CLIP-pretrained model ViT-B/32, which has a residual stream dimension of 768, and has 49 patches per image (224 / 32 = 7; 7 x 7 = 49). It will save the second-to-last layer (  layer -2 ). It will write 2.4M patches per shard, and save shards to a new directory  /local/scratch$USER/cache/saev .  note A note on storage space: A ViT-B/16 will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a  total of 723GB for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger. This script will also save a  metadata.json file that will record the relevant metadata for these activations, which will be read by future steps. The activations will be in  .bin files, numbered starting from 000000. To add your own models, see the guide to extending in  saev.activations .  Train SAEs on Activations To train an SAE, we need to specify: 1. Which activations to use as input. 2. SAE architectural stuff. 3. Optimization-related stuff.  The  saev.training module handles this. Run  uv run python -m saev train  help to see all the configuration. Continuing on from our example before, you might want to run something like:   uv run python -m saev train \\  data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \\  data.layer -2 \\  data.patches patches \\  data.no-scale-mean \\  data.no-scale-norm \\  sae.d-vit 768 \\  lr 5e-4     data. flags describe which activations to use.   data.shard-root should point to a directory with  .bin files and the  metadata.json file.   data.layer specifies the layer, and   data.patches says that want to train on individual patch activations, rather than the [CLS] token activation.   data.no-scale-mean and   data.no-scale-norm mean not to scale the activation mean or L2 norm. Anthropic's and OpenAI's papers suggest normalizing these factors, but  saev still has a bug with this, so I suggest not scaling these factors.   sae. flags are about the SAE itself.   sae.d-vit is the only one you need to change; the dimension of our ViT was 768 for a ViT-B, rather than the default of 1024 for a ViT-L. Finally, choose a slightly larger learning rate than the default with   lr 5e-4 . This will train one (1) sparse autoencoder on the data. See the section on sweeps to learn how to train multiple SAEs in parallel using only a single GPU.  Visualize the Learned Features Now that you've trained an SAE, you probably want to look at its learned features. One way to visualize an individual learned feature \\(f\\) is by picking out images that maximize the activation of feature \\(f\\). Since we train SAEs on patch-level activations, we try to find the top  patches for each feature \\(f\\). Then, we pick out the images those patches correspond to and create a heatmap based on SAE activation values.  note More advanced forms of visualization are possible (and valuable!), but should not be included in  saev unless they can be applied to every SAE/dataset combination. If you have specific visualizations, please add them to  contrib/ or another location.  saev.visuals records these maximally activating images for us. You can see all the options with  uv run python -m saev visuals  help . The most important configuration options: 1. The SAE checkpoint that you want to use (  ckpt ). 2. The ViT activations that you want to use (  data. options, should be roughly the same as the options you used to train your SAE, like the same layer, same   data.patches ). 3. The images that produced the ViT activations that you want to use ( images and   images. options, should be the same as what you used to generate your ViT activtions). 4. Some filtering options on which SAE latents to include (  log-freq-range ,   log-value-range ,   include-latents ,   n-latents ). Then, the script runs SAE inference on all of the ViT activations, calculates the images with maximal activation for each SAE feature, then retrieves the images from the original image dataset and highlights them for browsing later on.  note Because of limitations in the SAE training process, not all SAE latents (dimensions of \\(f\\ are equally interesting. Some latents are dead, some are  dense , some only fire on two images, etc. Typically, you want neurons that fire very strongly (high value) and fairly infrequently (low frequency). You might be interested in particular, fixed latents (  include-latents ).  I recommend using  saev.interactive.metrics to figure out good thresholds. So you might run:   uv run python -m saev visuals \\  ckpt checkpoints/abcdefg/sae.pt \\  dump-to /nfs/$USER/saev/webapp/abcdefg \\  data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \\  data.layer -2 \\  data.patches patches \\ images:imagenet-dataset   This will record the top 128 patches, and then save the unique images among those top 128 patches for each feature in the trained SAE. It will cache these best activations to disk, then start saving images to visualize later on.  saev.interactive.features is a small web application based on [marimo](https: marimo.io/) to interactively look at these images. You can run it with  uv run marimo edit saev/interactive/features.py .  Sweeps > tl;dr: basically the slow part of training SAEs is loading vit activations from disk, and since SAEs are pretty small compared to other models, you can train a bunch of different SAEs in parallel on the same data using a big GPU. That way you can sweep learning rate, lambda, etc. all on one GPU.  Why Parallel Sweeps SAE training optimizes for a unique bottleneck compared to typical ML workflows: disk I/O rather than GPU computation. When training on vision transformer activations, loading the pre-computed activation data from disk is often the slowest part of the process, not the SAE training itself. A single set of ImageNet activations for a vision transformer can require terabytes of storage. Reading this data repeatedly for each hyperparameter configuration would be extremely inefficient.  Parallelized Training Architecture To address this bottleneck, we implement parallel training that allows multiple SAE configurations to train simultaneously on the same data batch:  flowchart TD A[Pre-computed ViT Activations]  >|Slow I/O| B[Memory Buffer] B  >|Shared Batch| C[SAE Model 1] B  >|Shared Batch| D[SAE Model 2] B  >|Shared Batch| E[SAE Model 3] B  >|Shared Batch| F[ .]   import mermaid from 'https: cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';  This approach: - Loads each batch of activations  once from disk - Uses that same batch for multiple SAE models with different hyperparameters - Amortizes the slow I/O cost across all models in the sweep  Running a Sweep The  train command accepts a   sweep parameter that points to a TOML file defining the hyperparameter grid:   uv run python -m saev train  sweep configs/my_sweep.toml   Here's an example sweep configuration file:   [sae] sparsity_coeff = [1e-4, 2e-4, 3e-4] d_vit = 768 exp_factor = [8, 16] [data] scale_mean = true   This would train 6 models (3 sparsity coefficients \u00d7 2 expansion factors), each sharing the same data loading operation.  Limitations Not all parameters can be swept in parallel. Parameters that affect data loading (like  batch_size or dataset configuration) will cause the sweep to split into separate parallel groups. The system automatically handles this division to maximize efficiency.  Training Metrics and Visualizations When you train a sweep of SAEs, you probably want to understand which checkpoint is best.  saev provides some tools to help with that. First, we offer a tool to look at some basic summary statistics of all your trained checkpoints.  saev.interactive.metrics is a [marimo](https: marimo.io/) notebook (similar to Jupyter, but more interactive) for making L0 vs MSE plots by reading runs off of WandB. However, there are some pieces of code that need to be changed for you to use it.  todo Explain how to use the  saev.interactive.metrics notebook.  Need to change your wandb username from samuelstevens to USERNAME from wandb  Tag filter  Need to run the notebook on the same machine as the original ViT shards and the shards need to be there.  Think of better ways to do model and data keys  Look at examples  run visuals before features How to run visuals faster? explain how these features are visualized  How-To Guides  Reproduce To reproduce our findings from our preprint, you will need to train a couple SAEs on various datasets, then save visual examples so you can browse them in the notebooks.  Table of Contents 1. Save activations for ImageNet and iNat2021 for DINOv2, CLIP and BioCLIP. 2. Train SAEs on these activation datasets. 3. Pick the best SAE checkpoints for each combination. 4. Save visualizations for those best checkpoints.  Save Activations  Train SAEs  Choose Best Checkpoints  Save Visualizations Get visuals for the iNat-trained SAEs (BioCLIP and CLIP):   uv run python -m saev visuals \\  ckpt checkpoints/$CKPT/sae.pt \\  dump-to /$NFS/$USER/saev-visuals/$CKPT/ \\  log-freq-range -2.0 -1.0 \\  log-value-range -0.75 2.0 \\  data.shard-root /local/scratch/$USER/cache/saev/$SHARDS \\ images:image-folder-dataset \\  images.root /$NFS/$USER/datasets/inat21/train_mini/   Look at these visuals in the interactive notebook.   uv run marimo edit   Then open [localhost:2718](https: localhost:2718) in your browser and open the  saev/interactive/features.py file. Choose one of the checkpoints in the dropdown and click through the different neurons to find patterns in the underlying ViT.  Explanations  Related Work Various papers and internet posts on training SAEs for vision.  Preprints [An X-Ray Is Worth 15 Features: Sparse Autoencoders for Interpretable Radiology Report Generation](https: arxiv.org/pdf/2410.03334)  Haven't read this yet, but Hugo Fry is an author.  LessWrong [Towards Multimodal Interpretability: Learning Sparse Interpretable Features in Vision Transformers](https: www.lesswrong.com/posts/bCtbuWraqYTDtuARg/towards-multimodal-interpretability-learning-sparse-2)  Trains a sparse autoencoder on the 22nd layer of a CLIP ViT-L/14. First public work training an SAE on a ViT. Finds interesting features, demonstrating that SAEs work with ViTs. [Interpreting and Steering Features in Images](https: www.lesswrong.com/posts/Quqekpvx8BGMMcaem/interpreting-and-steering-features-in-images)  Havne't read it yet. [Case Study: Interpreting, Manipulating, and Controlling CLIP With Sparse Autoencoders](https: www.lesswrong.com/posts/iYFuZo9BMvr6GgMs5/case-study-interpreting-manipulating-and-controlling-clip)  Followup to the above work; haven't read it yet. [A Suite of Vision Sparse Autoencoders](https: www.lesswrong.com/posts/wrznNDMRmbQABAEMH/a-suite-of-vision-sparse-autoencoders)  Train a sparse autoencoder on various layers using the TopK with k=32 on a CLIP ViT-L/14 trained on LAION-2B. The SAE is trained on 1.2B tokens including patch (not just [CLS]). Limited evaluation.  Inference Instructions Briefly, you need to: 1. Download a checkpoint. 2. Get the code. 3. Load the checkpoint. 4. Get activations. Details are below.  Download a Checkpoint First, download an SAE checkpoint from the [Huggingface collection](https: huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195). For instance, you can choose the SAE trained on OpenAI's CLIP ViT-B/16 with ImageNet-1K activations [here](https: huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K). You can use  wget if you want:   wget https: huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K/resolve/main/sae.pt    Get the Code The easiest way to do this is to clone the code:   git clone https: github.com/OSU-NLP-Group/saev   You can also install the package from git if you use uv (not sure about pip or cuda):   uv add git+https: github.com/OSU-NLP-Group/saev   Or clone it and install it as an editable with pip, lik  pip install -e . in your virtual environment. Then you can do things like  from saev import  . .  note If you struggle to get  saev installed, open an issue on [GitHub](https: github.com/OSU-NLP-Group/saev) and I will figure out how to make it easier.  Load the Checkpoint   import saev.nn sae = saev.nn.load(\"PATH_TO_YOUR_SAE_CKPT.pt\")   Now you have a pretrained SAE.  Get Activations This is the hardest part. We need to: 1. Pass an image into a ViT 2. Record the dense ViT activations at the same layer that the SAE was trained on. 3. Pass the activations into the SAE to get sparse activations. 4. Do something interesting with the sparse SAE activations. There are examples of this in the demo code: for [classification](https: huggingface.co/spaces/samuelstevens/saev-image-classification/blob/main/app.py L318) and [semantic segmentation](https: huggingface.co/spaces/samuelstevens/saev-semantic-segmentation/blob/main/app.py L222). If the permalinks change, you are looking for the  get_sae_latents() functions in both files. Below is example code to do it using the  saev package.   import saev.nn import saev.activations img_transform = saev.activations.make_img_transform(\"clip\", \"ViT-B-16/openai\") vit = saev.activations.make_vit(\"clip\", \"ViT-B-16/openai\") recorded_vit = saev.activations.RecordedVisionTransformer(vit, 196, True, [10]) img = Image.open(\"example.jpg\") x = img_transform(img)  Add a batch dimension x = x[None,  .] _, vit_acts = recorded_vit(x)  Select the only layer in the batch and ignore the CLS token. vit_acts = vit_acts[:, 0, 1:, :] x_hat, f_x, loss = sae(vit_acts)   Now you have the reconstructed x ( x_hat ) and the sparse representation of all patches in the image ( f_x ). You might select the dimensions with maximal values for each patch and see what other images are maximimally activating.  todo Provide documentation for how get maximally activating images."
},
{
"ref":"saev.nn",
"url":1,
"doc":""
},
{
"ref":"saev.nn.SparseAutoencoder",
"url":1,
"doc":"Sparse auto-encoder (SAE) using L1 sparsity penalty. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.SparseAutoencoder.forward",
"url":1,
"doc":"Given x, calculates the reconstructed x_hat and the intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.decode",
"url":1,
"doc":"",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.init_b_dec",
"url":1,
"doc":"",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.normalize_w_dec",
"url":1,
"doc":"Set W_dec to unit-norm columns.",
"func":1
},
{
"ref":"saev.nn.SparseAutoencoder.remove_parallel_grads",
"url":1,
"doc":"Update grads so that they remove the parallel component (d_sae, d_vit) shape",
"func":1
},
{
"ref":"saev.nn.dump",
"url":1,
"doc":"Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https: docs.kidger.site/equinox/examples/serialisation). Arguments: fpath: filepath to save checkpoint to. sae: sparse autoencoder checkpoint to save.",
"func":1
},
{
"ref":"saev.nn.load",
"url":1,
"doc":"Loads a sparse autoencoder from disk.",
"func":1
},
{
"ref":"saev.nn.get_objective",
"url":1,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives",
"url":2,
"doc":"Uses [hypothesis]() and [hypothesis-torch](https: hypothesis-torch.readthedocs.io/en/stable/compatability/) to generate test cases to compare our normalized MSE implementation to a reference MSE implementation."
},
{
"ref":"saev.nn.test_objectives.test_mse_same",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.test_mse_zero_x_hat",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.test_mse_nonzero",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.test_safe_mse_large_x",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.test_factories",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.tensor_pair",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_objectives.test_safe_mse_hypothesis",
"url":2,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives",
"url":3,
"doc":""
},
{
"ref":"saev.nn.objectives.Loss",
"url":3,
"doc":"The loss term for an autoencoder training batch."
},
{
"ref":"saev.nn.objectives.Loss.loss",
"url":3,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.Loss.metrics",
"url":3,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.Objective",
"url":3,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.Objective.forward",
"url":3,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.VanillaLoss",
"url":3,
"doc":"The vanilla loss terms for an training batch."
},
{
"ref":"saev.nn.objectives.VanillaLoss.loss",
"url":3,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.VanillaLoss.metrics",
"url":3,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.VanillaLoss.l0",
"url":3,
"doc":"L0 magnitude of hidden activations."
},
{
"ref":"saev.nn.objectives.VanillaLoss.l1",
"url":3,
"doc":"L1 magnitude of hidden activations."
},
{
"ref":"saev.nn.objectives.VanillaLoss.mse",
"url":3,
"doc":"Reconstruction loss (mean squared error)."
},
{
"ref":"saev.nn.objectives.VanillaLoss.sparsity",
"url":3,
"doc":"Sparsity loss, typically lambda  L1."
},
{
"ref":"saev.nn.objectives.VanillaObjective",
"url":3,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.VanillaObjective.forward",
"url":3,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss",
"url":3,
"doc":"The composite loss terms for an training batch."
},
{
"ref":"saev.nn.objectives.MatryoshkaLoss.loss",
"url":3,
"doc":"Total loss."
},
{
"ref":"saev.nn.objectives.MatryoshkaObjective",
"url":3,
"doc":"Torch module for calculating the matryoshka loss for an SAE. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.objectives.MatryoshkaObjective.forward",
"url":3,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.nn.objectives.get_objective",
"url":3,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.ref_mean_squared_err",
"url":3,
"doc":"",
"func":1
},
{
"ref":"saev.nn.objectives.mean_squared_err",
"url":3,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_modeling",
"url":4,
"doc":""
},
{
"ref":"saev.nn.test_modeling.test_factories",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_modeling.relu_cfgs",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_modeling.test_sae_shapes",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_modeling.test_load_bioclip_checkpoint",
"url":4,
"doc":"",
"func":1
},
{
"ref":"saev.nn.test_modeling.test_dump_load_roundtrip",
"url":4,
"doc":"Write \u2192 load \u2192 verify state-dict & cfg equality.",
"func":1
},
{
"ref":"saev.nn.modeling",
"url":5,
"doc":"Neural network architectures for sparse autoencoders."
},
{
"ref":"saev.nn.modeling.SparseAutoencoder",
"url":5,
"doc":"Sparse auto-encoder (SAE) using L1 sparsity penalty. Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.forward",
"url":5,
"doc":"Given x, calculates the reconstructed x_hat and the intermediate activations f_x. Arguments: x: a batch of ViT activations.",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.decode",
"url":5,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.init_b_dec",
"url":5,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.normalize_w_dec",
"url":5,
"doc":"Set W_dec to unit-norm columns.",
"func":1
},
{
"ref":"saev.nn.modeling.SparseAutoencoder.remove_parallel_grads",
"url":5,
"doc":"Update grads so that they remove the parallel component (d_sae, d_vit) shape",
"func":1
},
{
"ref":"saev.nn.modeling.get_activation",
"url":5,
"doc":"",
"func":1
},
{
"ref":"saev.nn.modeling.dump",
"url":5,
"doc":"Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https: docs.kidger.site/equinox/examples/serialisation). Arguments: fpath: filepath to save checkpoint to. sae: sparse autoencoder checkpoint to save.",
"func":1
},
{
"ref":"saev.nn.modeling.load",
"url":5,
"doc":"Loads a sparse autoencoder from disk.",
"func":1
},
{
"ref":"saev.nn.modeling.current_git_commit",
"url":5,
"doc":"Best-effort short SHA of the repo containing  this file. Returns  None when   git executable is missing,  we\u2019re not inside a git repo (e.g. installed wheel),  or any git call errors out.",
"func":1
},
{
"ref":"saev.app",
"url":6,
"doc":""
},
{
"ref":"saev.app.data",
"url":7,
"doc":""
},
{
"ref":"saev.app.data.get_datasets",
"url":7,
"doc":"",
"func":1
},
{
"ref":"saev.app.data.get_img_raw",
"url":7,
"doc":"Get raw image and processed label from dataset. Returns: Tuple of Image.Image and classname.",
"func":1
},
{
"ref":"saev.app.data.to_sized",
"url":7,
"doc":"Convert raw vips image to standard model input size (resize + crop).",
"func":1
},
{
"ref":"saev.app.data.img_to_b64",
"url":7,
"doc":"",
"func":1
},
{
"ref":"saev.app.modeling",
"url":8,
"doc":""
},
{
"ref":"saev.app.modeling.Config",
"url":8,
"doc":"Configuration for a Vision Transformer (ViT) and Sparse Autoencoder (SAE) model pair. Stores paths and configuration needed to load and run a specific ViT+SAE combination."
},
{
"ref":"saev.app.modeling.Config.key",
"url":8,
"doc":"The lookup key."
},
{
"ref":"saev.app.modeling.Config.vit_family",
"url":8,
"doc":"The family of ViT model, e.g. 'clip' for CLIP models."
},
{
"ref":"saev.app.modeling.Config.vit_ckpt",
"url":8,
"doc":"Checkpoint identifier for the ViT model, either as HuggingFace path or model/checkpoint pair."
},
{
"ref":"saev.app.modeling.Config.sae_ckpt",
"url":8,
"doc":"Identifier for the SAE checkpoint to load."
},
{
"ref":"saev.app.modeling.Config.tensor_dpath",
"url":8,
"doc":"Directory containing precomputed tensors for this model combination."
},
{
"ref":"saev.app.modeling.Config.dataset_name",
"url":8,
"doc":"Which dataset to use."
},
{
"ref":"saev.app.modeling.Config.acts_cfg",
"url":8,
"doc":"Which activations to load for normalizing."
},
{
"ref":"saev.app.modeling.Config.wrapped_cfg",
"url":8,
"doc":""
},
{
"ref":"saev.app.modeling.get_model_lookup",
"url":8,
"doc":"",
"func":1
},
{
"ref":"saev.test_visuals",
"url":9,
"doc":""
},
{
"ref":"saev.test_visuals.test_gather_batched_small",
"url":9,
"doc":"",
"func":1
},
{
"ref":"saev.helpers",
"url":10,
"doc":"Useful helpers for  saev ."
},
{
"ref":"saev.helpers.get_cache_dir",
"url":10,
"doc":"Get cache directory from environment variables, defaulting to the current working directory (.) Returns: A path to a cache directory (might not exist yet).",
"func":1
},
{
"ref":"saev.helpers.progress",
"url":10,
"doc":"Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish. Args: it: Iterable to wrap. every: How many iterations between logging progress. desc: What to name the logger. total: If non-zero, how long the iterable is."
},
{
"ref":"saev.helpers.flattened",
"url":10,
"doc":"Flatten a potentially nested dict to a single-level dict with  . -separated keys.",
"func":1
},
{
"ref":"saev.helpers.get",
"url":10,
"doc":"",
"func":1
},
{
"ref":"saev.imaging",
"url":11,
"doc":""
},
{
"ref":"saev.imaging.add_highlights",
"url":11,
"doc":"",
"func":1
},
{
"ref":"saev.interactive",
"url":12,
"doc":""
},
{
"ref":"saev.interactive.metrics",
"url":13,
"doc":""
},
{
"ref":"saev.interactive.features",
"url":14,
"doc":""
},
{
"ref":"saev.visuals",
"url":15,
"doc":"There is some important notation used only in this file to dramatically shorten variable names. Variables suffixed with  _im refer to entire images, and variables suffixed with  _p refer to patches."
},
{
"ref":"saev.visuals.safe_load",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.visuals.gather_batched",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.visuals.GridElement",
"url":15,
"doc":"GridElement(img: PIL.Image.Image, label: str, patches: jaxtyping.Float[Tensor, 'n_patches'])"
},
{
"ref":"saev.visuals.GridElement.img",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.GridElement.label",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.GridElement.patches",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.make_img",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.visuals.get_new_topk",
"url":15,
"doc":"Picks out the new top k values among val1 and val2. Also keeps track of i1 and i2, then indices of the values in the original dataset. Args: val1: top k original SAE values. i1: the patch indices of those original top k values. val2: top k incoming SAE values. i2: the patch indices of those incoming top k values. k: k. Returns: The new top k values and their patch indices.",
"func":1
},
{
"ref":"saev.visuals.batched_idx",
"url":15,
"doc":"Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size. Args: total_size: total number of examples batch_size: maximum distance between the generated indices. Returns: A generator of (int, int) tuples that can slice up a list or a tensor.",
"func":1
},
{
"ref":"saev.visuals.get_sae_acts",
"url":15,
"doc":"Get SAE hidden layer activations for a batch of ViT activations. Args: vit_acts: Batch of ViT activations sae: Sparse autoencder. cfg: Experimental config.",
"func":1
},
{
"ref":"saev.visuals.TopKImg",
"url":15,
"doc":" todo Document this class."
},
{
"ref":"saev.visuals.TopKImg.top_values",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKImg.top_i",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKImg.mean_values",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKImg.sparsity",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKImg.distributions",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKImg.percentiles",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.get_topk_img",
"url":15,
"doc":"Gets the top k images for each latent in the SAE. The top k images are for latent i are sorted by max over all images: f_x(cls)[i] Thus, we will never have duplicate images for a given latent. But we also will not have patch-level activations (a nice heatmap). Args: cfg: Config. Returns: A tuple of TopKImg and the first m features' activation distributions.",
"func":1
},
{
"ref":"saev.visuals.TopKPatch",
"url":15,
"doc":" todo Document this class."
},
{
"ref":"saev.visuals.TopKPatch.top_values",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKPatch.top_i",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKPatch.mean_values",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKPatch.sparsity",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKPatch.distributions",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.TopKPatch.percentiles",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.get_topk_patch",
"url":15,
"doc":"Gets the top k images for each latent in the SAE. The top k images are for latent i are sorted by max over all patches: f_x(patch)[i] Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent. Args: cfg: Config. Returns: A tuple of TopKPatch and m randomly sampled activation distributions.",
"func":1
},
{
"ref":"saev.visuals.dump_activations",
"url":15,
"doc":"For each SAE latent, we want to know which images have the most total \"activation\". That is, we keep track of each patch",
"func":1
},
{
"ref":"saev.visuals.plot_activation_distributions",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.visuals.main",
"url":15,
"doc":" todo document this function. Dump top-k images to a directory. Args: cfg: Configuration object.",
"func":1
},
{
"ref":"saev.visuals.PercentileEstimator",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.PercentileEstimator.update",
"url":15,
"doc":"Update the estimator with a new value. This method maintains the marker positions using the P2 algorithm rules. When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions. Arguments: x: The new value to incorporate into the estimation",
"func":1
},
{
"ref":"saev.visuals.PercentileEstimator.estimate",
"url":15,
"doc":""
},
{
"ref":"saev.visuals.test_online_quantile_estimation",
"url":15,
"doc":"",
"func":1
},
{
"ref":"saev.test_config",
"url":16,
"doc":""
},
{
"ref":"saev.test_config.test_expand",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_expand_two_fields",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_expand_nested",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_expand_nested_and_unnested",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_expand_nested_and_unnested_backwards",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_expand_multiple",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.test_config.test_union_is_exhaustive",
"url":16,
"doc":"",
"func":1
},
{
"ref":"saev.activations",
"url":17,
"doc":"To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one. This module handles that additional complexity. Conceptually, activations are either thought of as 1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a  dataset 2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations."
},
{
"ref":"saev.activations.RecordedVisionTransformer",
"url":17,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.activations.RecordedVisionTransformer.hook",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.RecordedVisionTransformer.reset",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.RecordedVisionTransformer.activations",
"url":17,
"doc":""
},
{
"ref":"saev.activations.RecordedVisionTransformer.forward",
"url":17,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.activations.Clip",
"url":17,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.activations.Clip.get_residuals",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Clip.get_patches",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Clip.forward",
"url":17,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.activations.Siglip",
"url":17,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.activations.Siglip.get_residuals",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Siglip.get_patches",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Siglip.forward",
"url":17,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.activations.DinoV2",
"url":17,
"doc":"Base class for all neural network modules. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes import torch.nn as nn import torch.nn.functional as F class Model(nn.Module): def __init__(self) -> None: super().__init__() self.conv1 = nn.Conv2d(1, 20, 5) self.conv2 = nn.Conv2d(20, 20, 5) def forward(self, x): x = F.relu(self.conv1(x return F.relu(self.conv2(x Submodules assigned in this way will be registered, and will have their parameters converted too when you call :meth: to , etc.  note As per the example above, an  __init__() call to the parent class must be made before assignment on the child. :ivar training: Boolean represents whether this module is in training or evaluation mode. :vartype training: bool Initialize internal Module state, shared by both nn.Module and ScriptModule."
},
{
"ref":"saev.activations.DinoV2.get_residuals",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.DinoV2.get_patches",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.DinoV2.forward",
"url":17,
"doc":"Define the computation performed at every call. Should be overridden by all subclasses.  note Although the recipe for forward pass needs to be defined within this function, one should call the :class: Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.",
"func":1
},
{
"ref":"saev.activations.make_vit",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.make_img_transform",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Dataset",
"url":17,
"doc":"Dataset of activations from disk."
},
{
"ref":"saev.activations.Dataset.cfg",
"url":17,
"doc":"Configuration; set via CLI args."
},
{
"ref":"saev.activations.Dataset.metadata",
"url":17,
"doc":"Activations metadata; automatically loaded from disk."
},
{
"ref":"saev.activations.Dataset.layer_index",
"url":17,
"doc":"Layer index into the shards if we are choosing a specific layer."
},
{
"ref":"saev.activations.Dataset.scalar",
"url":17,
"doc":"Normalizing scalar such that  x / scalar  _2 ~= sqrt(d_vit)."
},
{
"ref":"saev.activations.Dataset.act_mean",
"url":17,
"doc":"Mean activation."
},
{
"ref":"saev.activations.Dataset.Example",
"url":17,
"doc":"Individual example."
},
{
"ref":"saev.activations.Dataset.transform",
"url":17,
"doc":"Apply a scalar normalization so the mean squared L2 norm is same as d_vit. This is from 'Scaling Monosemanticity': > As a preprocessing step we apply a scalar normalization to the model activations so their average squared L2 norm is the residual stream dimension So we divide by self.scalar which is the datasets (approximate) L2 mean before normalization divided by sqrt(d_vit).",
"func":1
},
{
"ref":"saev.activations.Dataset.d_vit",
"url":17,
"doc":"Dimension of the underlying vision transformer's embedding space."
},
{
"ref":"saev.activations.Dataset.get_shard_patches",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Dataset.get_img_patches",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.setup",
"url":17,
"doc":"Run dataset-specific setup. These setup functions can assume they are the only job running, but they should be idempotent; they should be safe (and ideally cheap) to run multiple times in a row.",
"func":1
},
{
"ref":"saev.activations.setup_imagenet",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.setup_imagefolder",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.setup_ade20k",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.get_dataset",
"url":17,
"doc":"Gets the dataset for the current experiment; delegates construction to dataset-specific functions. Args: cfg: Experiment config. img_transform: Image transform to be applied to each image. Returns: A dataset that has dictionaries with  'image' ,  'index' ,  'target' , and  'label' keys containing examples.",
"func":1
},
{
"ref":"saev.activations.get_dataloader",
"url":17,
"doc":"Gets the dataloader for the current experiment; delegates dataloader construction to dataset-specific functions. Args: cfg: Experiment config. img_transform: Image transform to be applied to each image. Returns: A PyTorch Dataloader that yields dictionaries with  'image' keys containing image batches.",
"func":1
},
{
"ref":"saev.activations.get_default_dataloader",
"url":17,
"doc":"Get a dataloader for a default map-style dataset. Args: cfg: Config. img_transform: Image transform to be applied to each image. Returns: A PyTorch Dataloader that yields dictionaries with  'image' keys containing image batches,  'index' keys containing original dataset indices and  'label' keys containing label batches.",
"func":1
},
{
"ref":"saev.activations.Imagenet",
"url":17,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.activations.ImageFolder",
"url":17,
"doc":"A generic data loader where the images are arranged in this way by default:  root/dog/xxx.png root/dog/xxy.png root/dog/[ .]/xxz.png root/cat/123.png root/cat/nsdf3.png root/cat/[ .]/asd932_.png This class inherits from :class: ~torchvision.datasets.DatasetFolder so the same methods can be overridden to customize the dataset. Args: root (str or  pathlib.Path ): Root directory path. transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g,  transforms.RandomCrop target_transform (callable, optional): A function/transform that takes in the target and transforms it. loader (callable, optional): A function to load an image given its path. is_valid_file (callable, optional): A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files) allow_empty(bool, optional): If True, empty folders are considered to be valid classes. An error is raised on empty folders if False (default). Attributes: classes (list): List of the class names sorted alphabetically. class_to_idx (dict): Dict with items (class_name, class_index). imgs (list): List of (image path, class_index) tuples"
},
{
"ref":"saev.activations.Ade20k",
"url":17,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.activations.Ade20k.samples",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Ade20k.Sample",
"url":17,
"doc":""
},
{
"ref":"saev.activations.main",
"url":17,
"doc":"Args: cfg: Config for activations.",
"func":1
},
{
"ref":"saev.activations.worker_fn",
"url":17,
"doc":"Args: cfg: Config for activations.",
"func":1
},
{
"ref":"saev.activations.ShardWriter",
"url":17,
"doc":"ShardWriter is a stateful object that handles sharded activation writing to disk."
},
{
"ref":"saev.activations.ShardWriter.root",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.shape",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.shard",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.acts_path",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.acts",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.filled",
"url":17,
"doc":""
},
{
"ref":"saev.activations.ShardWriter.flush",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.ShardWriter.next_shard",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Metadata",
"url":17,
"doc":"Metadata(vit_family: str, vit_ckpt: str, layers: tuple[int,  .], n_patches_per_img: int, cls_token: bool, d_vit: int, seed: int, n_imgs: int, n_patches_per_shard: int, data: str)"
},
{
"ref":"saev.activations.Metadata.vit_family",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.vit_ckpt",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.layers",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.n_patches_per_img",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.cls_token",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.d_vit",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.seed",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.n_imgs",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.n_patches_per_shard",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.data",
"url":17,
"doc":""
},
{
"ref":"saev.activations.Metadata.from_cfg",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Metadata.load",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Metadata.dump",
"url":17,
"doc":"",
"func":1
},
{
"ref":"saev.activations.Metadata.hash",
"url":17,
"doc":""
},
{
"ref":"saev.activations.get_acts_dir",
"url":17,
"doc":"Return the activations directory based on the relevant values of a config. Also saves a metadata.json file to that directory for human reference. Args: cfg: Config for experiment. Returns: Directory to where activations should be dumped/loaded from.",
"func":1
},
{
"ref":"saev.config",
"url":18,
"doc":"All configs for all saev jobs.  Import Times This module should be very fast to import so that  python main.py  help is fast. This means that the top-level imports should not include big packages like numpy, torch, etc. For example,  TreeOfLife.n_imgs imports numpy when it's needed, rather than importing it at the top level. Also contains code for expanding configs with lists into lists of configs (grid search). Might be expanded in the future to support pseudo-random sampling from distributions to support random hyperparameter search, as in [this file](https: github.com/samuelstevens/sax/blob/main/sax/sweep.py)."
},
{
"ref":"saev.config.ImagenetDataset",
"url":18,
"doc":"Configuration for HuggingFace Imagenet."
},
{
"ref":"saev.config.ImagenetDataset.n_imgs",
"url":18,
"doc":"Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."
},
{
"ref":"saev.config.ImagenetDataset.name",
"url":18,
"doc":"Dataset name on HuggingFace. Don't need to change this "
},
{
"ref":"saev.config.ImagenetDataset.split",
"url":18,
"doc":"Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."
},
{
"ref":"saev.config.ImageFolderDataset",
"url":18,
"doc":"Configuration for a generic image folder dataset."
},
{
"ref":"saev.config.ImageFolderDataset.n_imgs",
"url":18,
"doc":"Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."
},
{
"ref":"saev.config.ImageFolderDataset.root",
"url":18,
"doc":"Where the class folders with images are stored."
},
{
"ref":"saev.config.Ade20kDataset",
"url":18,
"doc":""
},
{
"ref":"saev.config.Ade20kDataset.n_imgs",
"url":18,
"doc":""
},
{
"ref":"saev.config.Ade20kDataset.root",
"url":18,
"doc":"Where the class folders with images are stored."
},
{
"ref":"saev.config.Ade20kDataset.split",
"url":18,
"doc":"Data split."
},
{
"ref":"saev.config.Activations",
"url":18,
"doc":"Configuration for calculating and saving ViT activations."
},
{
"ref":"saev.config.Activations.cls_token",
"url":18,
"doc":"Whether the model has a [CLS] token."
},
{
"ref":"saev.config.Activations.d_vit",
"url":18,
"doc":"Dimension of the ViT activations (depends on model)."
},
{
"ref":"saev.config.Activations.data",
"url":18,
"doc":"Which dataset to use."
},
{
"ref":"saev.config.Activations.device",
"url":18,
"doc":"Which device to use."
},
{
"ref":"saev.config.Activations.dump_to",
"url":18,
"doc":"Where to write shards."
},
{
"ref":"saev.config.Activations.log_to",
"url":18,
"doc":"Where to log Slurm job stdout/stderr."
},
{
"ref":"saev.config.Activations.n_patches_per_img",
"url":18,
"doc":"Number of ViT patches per image (depends on model)."
},
{
"ref":"saev.config.Activations.n_patches_per_shard",
"url":18,
"doc":"Number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."
},
{
"ref":"saev.config.Activations.n_workers",
"url":18,
"doc":"Number of dataloader workers."
},
{
"ref":"saev.config.Activations.seed",
"url":18,
"doc":"Random seed."
},
{
"ref":"saev.config.Activations.slurm",
"url":18,
"doc":"Whether to use  submitit to run jobs on a Slurm cluster."
},
{
"ref":"saev.config.Activations.slurm_acct",
"url":18,
"doc":"Slurm account string."
},
{
"ref":"saev.config.Activations.ssl",
"url":18,
"doc":"Whether to use SSL."
},
{
"ref":"saev.config.Activations.vit_batch_size",
"url":18,
"doc":"Batch size for ViT inference."
},
{
"ref":"saev.config.Activations.vit_ckpt",
"url":18,
"doc":"Specific model checkpoint."
},
{
"ref":"saev.config.Activations.vit_family",
"url":18,
"doc":"Which model family."
},
{
"ref":"saev.config.Activations.vit_layers",
"url":18,
"doc":"Which layers to save. By default, the second-to-last layer."
},
{
"ref":"saev.config.DataLoad",
"url":18,
"doc":"Configuration for loading activation data from disk."
},
{
"ref":"saev.config.DataLoad.clamp",
"url":18,
"doc":"Maximum value for activations; activations will be clamped to within [-clamp, clamp] ."
},
{
"ref":"saev.config.DataLoad.layer",
"url":18,
"doc":" todo: document this field."
},
{
"ref":"saev.config.DataLoad.n_random_samples",
"url":18,
"doc":"Number of random samples used to calculate approximate dataset means at startup."
},
{
"ref":"saev.config.DataLoad.patches",
"url":18,
"doc":"Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return all patches. 'meanpool' returns the mean of all image patches."
},
{
"ref":"saev.config.DataLoad.scale_mean",
"url":18,
"doc":"Whether to subtract approximate dataset means from examples. If a string, manually load from the filepath."
},
{
"ref":"saev.config.DataLoad.scale_norm",
"url":18,
"doc":"Whether to scale average dataset norm to sqrt(d_vit). If a string, manually load from the filepath."
},
{
"ref":"saev.config.DataLoad.shard_root",
"url":18,
"doc":"Directory with .bin shards and a metadata.json file."
},
{
"ref":"saev.config.Relu",
"url":18,
"doc":"Relu(d_vit: int = 1024, exp_factor: int = 16, n_reinit_samples: int = 524288, remove_parallel_grads: bool = True, normalize_w_dec: bool = True, seed: int = 0)"
},
{
"ref":"saev.config.Relu.d_sae",
"url":18,
"doc":""
},
{
"ref":"saev.config.Relu.d_vit",
"url":18,
"doc":""
},
{
"ref":"saev.config.Relu.exp_factor",
"url":18,
"doc":"Expansion factor for SAE."
},
{
"ref":"saev.config.Relu.n_reinit_samples",
"url":18,
"doc":"Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-bias. We use the regular mean."
},
{
"ref":"saev.config.Relu.normalize_w_dec",
"url":18,
"doc":"Whether to make sure W_dec has unit norm columns. See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder for original citation."
},
{
"ref":"saev.config.Relu.remove_parallel_grads",
"url":18,
"doc":"Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https: transformer-circuits.pub/2023/monosemantic-features/index.html appendix-autoencoder-optimization for the original discussion from Anthropic."
},
{
"ref":"saev.config.Relu.seed",
"url":18,
"doc":"Random seed."
},
{
"ref":"saev.config.JumpRelu",
"url":18,
"doc":"Implementation of the JumpReLU activation function for SAEs. Not implemented."
},
{
"ref":"saev.config.Vanilla",
"url":18,
"doc":"Vanilla(sparsity_coeff: float = 0.0004)"
},
{
"ref":"saev.config.Vanilla.sparsity_coeff",
"url":18,
"doc":"How much to weight sparsity loss term."
},
{
"ref":"saev.config.Matryoshka",
"url":18,
"doc":"Config for the Matryoshka loss for another arbitrary SAE class. Reference code is here: https: github.com/noanabeshima/matryoshka-saes and the original reading is https: sparselatents.com/matryoshka.html and https: arxiv.org/pdf/2503.17547."
},
{
"ref":"saev.config.Matryoshka.n_prefixes",
"url":18,
"doc":"Number of random length prefixes to use for loss calculation."
},
{
"ref":"saev.config.Train",
"url":18,
"doc":"Configuration for training a sparse autoencoder on a vision transformer."
},
{
"ref":"saev.config.Train.ckpt_path",
"url":18,
"doc":"Where to save checkpoints."
},
{
"ref":"saev.config.Train.data",
"url":18,
"doc":"Data configuration"
},
{
"ref":"saev.config.Train.device",
"url":18,
"doc":"Hardware device."
},
{
"ref":"saev.config.Train.log_every",
"url":18,
"doc":"How often to log to WandB."
},
{
"ref":"saev.config.Train.log_to",
"url":18,
"doc":"Where to log Slurm job stdout/stderr."
},
{
"ref":"saev.config.Train.lr",
"url":18,
"doc":"Learning rate."
},
{
"ref":"saev.config.Train.n_lr_warmup",
"url":18,
"doc":"Number of learning rate warmup steps."
},
{
"ref":"saev.config.Train.n_patches",
"url":18,
"doc":"Number of SAE training examples."
},
{
"ref":"saev.config.Train.n_sparsity_warmup",
"url":18,
"doc":"Number of sparsity coefficient warmup steps."
},
{
"ref":"saev.config.Train.n_workers",
"url":18,
"doc":"Number of dataloader workers."
},
{
"ref":"saev.config.Train.objective",
"url":18,
"doc":"SAE loss configuration."
},
{
"ref":"saev.config.Train.sae",
"url":18,
"doc":"SAE configuration."
},
{
"ref":"saev.config.Train.sae_batch_size",
"url":18,
"doc":"Batch size for SAE training."
},
{
"ref":"saev.config.Train.seed",
"url":18,
"doc":"Random seed."
},
{
"ref":"saev.config.Train.slurm",
"url":18,
"doc":"Whether to use  submitit to run jobs on a Slurm cluster."
},
{
"ref":"saev.config.Train.slurm_acct",
"url":18,
"doc":"Slurm account string."
},
{
"ref":"saev.config.Train.tag",
"url":18,
"doc":"Tag to add to WandB run."
},
{
"ref":"saev.config.Train.track",
"url":18,
"doc":"Whether to track with WandB."
},
{
"ref":"saev.config.Train.wandb_project",
"url":18,
"doc":"WandB project name."
},
{
"ref":"saev.config.Visuals",
"url":18,
"doc":"Configuration for generating visuals from trained SAEs."
},
{
"ref":"saev.config.Visuals.root",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.top_values_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.top_img_i_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.top_patch_i_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.mean_values_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.sparsity_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.distributions_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.percentiles_fpath",
"url":18,
"doc":""
},
{
"ref":"saev.config.Visuals.ckpt",
"url":18,
"doc":"Path to the sae.pt file."
},
{
"ref":"saev.config.Visuals.data",
"url":18,
"doc":"Data configuration."
},
{
"ref":"saev.config.Visuals.device",
"url":18,
"doc":"Which accelerator to use."
},
{
"ref":"saev.config.Visuals.dump_to",
"url":18,
"doc":"Where to save data."
},
{
"ref":"saev.config.Visuals.epsilon",
"url":18,
"doc":"Value to add to avoid log(0)."
},
{
"ref":"saev.config.Visuals.images",
"url":18,
"doc":"Which images to use."
},
{
"ref":"saev.config.Visuals.include_latents",
"url":18,
"doc":"Latents to always include, no matter what."
},
{
"ref":"saev.config.Visuals.log_freq_range",
"url":18,
"doc":"Log10 frequency range for which to save images."
},
{
"ref":"saev.config.Visuals.log_value_range",
"url":18,
"doc":"Log10 frequency range for which to save images."
},
{
"ref":"saev.config.Visuals.n_distributions",
"url":18,
"doc":"Number of features to save distributions for."
},
{
"ref":"saev.config.Visuals.n_latents",
"url":18,
"doc":"Maximum number of latents to save images for."
},
{
"ref":"saev.config.Visuals.n_workers",
"url":18,
"doc":"Number of dataloader workers."
},
{
"ref":"saev.config.Visuals.percentile",
"url":18,
"doc":"Percentile to estimate for outlier detection."
},
{
"ref":"saev.config.Visuals.sae_batch_size",
"url":18,
"doc":"Batch size for SAE inference."
},
{
"ref":"saev.config.Visuals.seed",
"url":18,
"doc":"Random seed."
},
{
"ref":"saev.config.Visuals.sort_by",
"url":18,
"doc":"How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."
},
{
"ref":"saev.config.Visuals.top_k",
"url":18,
"doc":"How many images per SAE feature to store."
},
{
"ref":"saev.config.Visuals.topk_batch_size",
"url":18,
"doc":"Number of examples to apply top-k op to."
},
{
"ref":"saev.config.grid",
"url":18,
"doc":"",
"func":1
},
{
"ref":"saev.config.expand",
"url":18,
"doc":"Expands dicts with (nested) lists into a list of (nested) dicts.",
"func":1
},
{
"ref":"saev.colors",
"url":19,
"doc":""
},
{
"ref":"saev.training",
"url":20,
"doc":"Trains many SAEs in parallel to amortize the cost of loading a single batch of data over many SAE training runs."
},
{
"ref":"saev.training.init_b_dec_batched",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.make_saes",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.ParallelWandbRun",
"url":20,
"doc":"Inspired by https: community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/3."
},
{
"ref":"saev.training.ParallelWandbRun.log",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.ParallelWandbRun.finish",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.main",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.train",
"url":20,
"doc":"Explicitly declare the optimizer, schedulers, dataloader, etc outside of  main so that all the variables are dropped from scope and can be garbage collected.",
"func":1
},
{
"ref":"saev.training.EvalMetrics",
"url":20,
"doc":"Results of evaluating a trained SAE on a datset."
},
{
"ref":"saev.training.EvalMetrics.l0",
"url":20,
"doc":"Mean L0 across all examples."
},
{
"ref":"saev.training.EvalMetrics.l1",
"url":20,
"doc":"Mean L1 across all examples."
},
{
"ref":"saev.training.EvalMetrics.mse",
"url":20,
"doc":"Mean MSE across all examples."
},
{
"ref":"saev.training.EvalMetrics.n_dead",
"url":20,
"doc":"Number of neurons that never fired on any example."
},
{
"ref":"saev.training.EvalMetrics.n_almost_dead",
"url":20,
"doc":"Number of neurons that fired on fewer than  almost_dead_threshold of examples."
},
{
"ref":"saev.training.EvalMetrics.n_dense",
"url":20,
"doc":"Number of neurons that fired on more than  dense_threshold of examples."
},
{
"ref":"saev.training.EvalMetrics.freqs",
"url":20,
"doc":"How often each feature fired."
},
{
"ref":"saev.training.EvalMetrics.mean_values",
"url":20,
"doc":"The mean value for each feature when it did fire."
},
{
"ref":"saev.training.EvalMetrics.almost_dead_threshold",
"url":20,
"doc":"Threshold for an \"almost dead\" neuron."
},
{
"ref":"saev.training.EvalMetrics.dense_threshold",
"url":20,
"doc":"Threshold for a dense neuron."
},
{
"ref":"saev.training.EvalMetrics.for_wandb",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.evaluate",
"url":20,
"doc":"Evaluates SAE quality by counting the number of dead features and the number of dense features. Also makes histogram plots to help human qualitative comparison.  todo Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.",
"func":1
},
{
"ref":"saev.training.BatchLimiter",
"url":20,
"doc":"Limits the number of batches to only return  n_samples total samples."
},
{
"ref":"saev.training.split_cfgs",
"url":20,
"doc":"Splits configs into groups that can be parallelized. Arguments: A list of configs from a sweep file. Returns: A list of lists, where the configs in each sublist do not differ in any keys that are in  CANNOT_PARALLELIZE . This means that each sublist is a valid \"parallel\" set of configs for  train .",
"func":1
},
{
"ref":"saev.training.make_hashable",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.Scheduler",
"url":20,
"doc":""
},
{
"ref":"saev.training.Scheduler.step",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.training.Warmup",
"url":20,
"doc":"Linearly increases from  init to  final over  n_warmup_steps steps."
},
{
"ref":"saev.training.Warmup.step",
"url":20,
"doc":"",
"func":1
},
{
"ref":"saev.test_training",
"url":21,
"doc":""
},
{
"ref":"saev.test_training.test_split_cfgs_on_single_key",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.test_training.test_split_cfgs_on_single_key_with_multiple_per_key",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.test_training.test_split_cfgs_on_multiple_keys_with_multiple_per_key",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.test_training.test_split_cfgs_no_bad_keys",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.test_training.DummyDS",
"url":21,
"doc":"An abstract class representing a :class: Dataset . All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite :meth: __getitem__ , supporting fetching a data sample for a given key. Subclasses could also optionally overwrite :meth: __len__ , which is expected to return the size of the dataset by many :class: ~torch.utils.data.Sampler implementations and the default options of :class: ~torch.utils.data.DataLoader . Subclasses could also optionally implement :meth: __getitems__ , for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.  note :class: ~torch.utils.data.DataLoader by default constructs an index sampler that yields integral indices. To make it work with a map-style dataset with non-integral indices/keys, a custom sampler must be provided."
},
{
"ref":"saev.test_training.test_one_training_step",
"url":21,
"doc":"",
"func":1
},
{
"ref":"saev.test_training.test_one_training_step_matryoshka",
"url":21,
"doc":"A minimal end-to-end training-loop smoke test for the Matryoshka objective.",
"func":1
},
{
"ref":"saev.test_activations",
"url":22,
"doc":"Test that the cached activations are actually correct. These tests are quite slow"
},
{
"ref":"saev.test_activations.test_dataloader_batches",
"url":22,
"doc":"",
"func":1
},
{
"ref":"saev.test_activations.test_shard_writer_and_dataset_e2e",
"url":22,
"doc":"",
"func":1
}
]