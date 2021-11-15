
<!DOCTYPE html>

<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ImageNet Project &#8212; ImageNet Project</title>
    
  <link href="_static/css/theme.css" rel="stylesheet" />
  <link href="_static/css/index.c5995385ac14fb8791e8eb36b4908be2.css" rel="stylesheet" />

    
  <link rel="stylesheet"
    href="_static/vendor/fontawesome/5.13.0/css/all.min.css">
  <link rel="preload" as="font" type="font/woff2" crossorigin
    href="_static/vendor/fontawesome/5.13.0/webfonts/fa-solid-900.woff2">
  <link rel="preload" as="font" type="font/woff2" crossorigin
    href="_static/vendor/fontawesome/5.13.0/webfonts/fa-brands-400.woff2">

    
      

    
    <link rel="stylesheet" type="text/css" href="_static/pygments.css" />
    <link rel="stylesheet" type="text/css" href="_static/basic.css" />
    
  <link rel="preload" as="script" href="_static/js/index.1c5a1a01449ed65a7b51.js">

    <script data-url_root="./" id="documentation_options" src="_static/documentation_options.js"></script>
    <script src="_static/jquery.js"></script>
    <script src="_static/underscore.js"></script>
    <script src="_static/doctools.js"></script>
    <script crossorigin="anonymous" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>
    <link rel="shortcut icon" href="_static/favicon.ico"/>
    <link rel="index" title="Index" href="genindex.html" />
    <link rel="search" title="Search" href="search.html" />
    <link rel="next" title="Base Classes and Functions" href="defs.html" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="docsearch:language" content="en" />
    
  </head>
  <body data-spy="scroll" data-target="#bd-toc-nav" data-offset="80">
    
    <div class="container-fluid" id="banner"></div>

    
    <nav class="navbar navbar-light navbar-expand-lg bg-light fixed-top bd-navbar" id="navbar-main"><div class="container-xl">

  <div id="navbar-start">
    
    
<a class="navbar-brand" href="#">
<p class="title">ImageNet</p>
</a>

    
  </div>

  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbar-collapsible" aria-controls="navbar-collapsible" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>

  
  <div id="navbar-collapsible" class="col-lg-9 collapse navbar-collapse">
    <div id="navbar-center" class="mr-auto">
      
      <div class="navbar-center-item">
        <ul id="navbar-main-elements" class="navbar-nav">
    <li class="toctree-l1 nav-item">
 <a class="reference internal nav-link" href="defs.html">
  Base Classes and Functions
 </a>
</li>

<li class="toctree-l1 nav-item">
 <a class="reference internal nav-link" href="datasets.html">
  Dataset Classes
 </a>
</li>

<li class="toctree-l1 nav-item">
 <a class="reference internal nav-link" href="nets.html">
  Neural Networks
 </a>
</li>

<li class="toctree-l1 nav-item">
 <a class="reference internal nav-link" href="decorators.html">
  Decorators
 </a>
</li>

    
</ul>
      </div>
      
    </div>

    <div id="navbar-end">
      
      <div class="navbar-end-item">
        <ul id="navbar-icon-links" class="navbar-nav" aria-label="Icon Links">
      </ul>
      </div>
      
    </div>
  </div>
</div>
    </nav>
    

    <div class="container-xl">
      <div class="row">
          
            
            <!-- Only show if we have sidebars configured, else just a small margin  -->
            <div class="col-12 col-md-3 bd-sidebar"><form class="bd-search d-flex align-items-center" action="search.html" method="get">
  <i class="icon fas fa-search"></i>
  <input type="search" class="form-control" name="q" id="search-input" placeholder="Search the docs ..." aria-label="Search the docs ..." autocomplete="off" >
</form><nav class="bd-links" id="bd-docs-nav" aria-label="Main navigation">
  <div class="bd-toc-item active">
    
  </div>
</nav>
            </div>
            
          

          
          <div class="d-none d-xl-block col-xl-2 bd-toc">
            
              
              <div class="toc-item">
                
<div class="tocsection onthispage pt-5 pb-3">
    <i class="fas fa-list"></i> On this page
</div>

<nav id="bd-toc-nav">
    <ul class="visible nav section-nav flex-column">
 <li class="toc-h1 nav-item toc-entry">
  <a class="reference internal nav-link" href="#">
   ImageNet Project
  </a>
  <ul class="visible nav section-nav flex-column">
   <li class="toc-h2 nav-item toc-entry">
    <a class="reference internal nav-link" href="#introduction">
     Introduction
    </a>
   </li>
   <li class="toc-h2 nav-item toc-entry">
    <a class="reference internal nav-link" href="#dataset">
     Dataset
    </a>
   </li>
   <li class="toc-h2 nav-item toc-entry">
    <a class="reference internal nav-link" href="#current-state-of-development">
     Current State of Development
    </a>
   </li>
   <li class="toc-h2 nav-item toc-entry">
    <a class="reference internal nav-link" href="#objectives">
     Objectives
    </a>
   </li>
  </ul>
 </li>
 <li class="toc-h1 nav-item toc-entry">
  <a class="reference internal nav-link" href="#indices-and-tables">
   Indices and tables
  </a>
 </li>
</ul>

</nav>
              </div>
              
              <div class="toc-item">
                
              </div>
              
            
          </div>
          

          
          
            
          
          <main class="col-12 col-md-9 col-xl-7 py-md-5 pl-md-5 pr-md-4 bd-content" role="main">
              
              <div>
                
  <div class="section" id="imagenet-project">
<h1>ImageNet Project<a class="headerlink" href="#imagenet-project" title="Permalink to this headline">¶</a></h1>
<div class="section" id="introduction">
<h2>Introduction<a class="headerlink" href="#introduction" title="Permalink to this headline">¶</a></h2>
<p>Despite having previous experiences in developing CNNs in PyTorch, I’ve always
felt overwhelmed by the number of different of ways that one can combine
convolutional, max-pooling and linear layers - with different kernel sizes and
padding sizes, strides, dilation and feature numbers.</p>
<p>My objective is to develop skills in python and get used to using GitHub while
exploring a variety of CNN architectures to find in practice which ones work
best for different applications. This project will also serve as a portfolio.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>This is a <em>work in progress</em>.</p>
</div>
</div>
<div class="section" id="dataset">
<h2>Dataset<a class="headerlink" href="#dataset" title="Permalink to this headline">¶</a></h2>
<p>The first dataset I’ve chosen to use is the
<a class="reference external" href="https://arxiv.org/abs/1701.08380">HASYv2 dataset</a>, because it has many more
classes and symbols than other symbol recognition datasets such as MNIST, and
the final models could possibly be adapted in the future for translating
handwritten equations (even if they are handwritten through a mouse pad of
sorts) into LaTeX equations.</p>
<p>This also inspires me to develop some kind of application where the user can
draw symbols in a 32x32 pixel grid with its mouse, and a trained net will try to
guess it at the same time. The user can then add it live to the LaTeX equation.
This application idea draws inspiration from Google’s
<a class="reference external" href="https://quickdraw.withgoogle.com/">Quick, Draw!</a>.</p>
<p>Update 14/11/2021: I’ve found out through a friend that a website already exists
for this: <a class="reference external" href="http://detexify.kirelabs.org/symbols.html">http://detexify.kirelabs.org/symbols.html</a>. Still, my development
continues.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>In order to work with the dataset on
<a class="reference external" href="https://colab.research.google.com/">GoogleColab</a>, I’ve tried uploading
the data to GoogleDrive, which didn’t work because it has trouble dealing
with such a big amount of data (even unzipping the files directly from Colab
didn’t work). Because of this, I’ve organized them in a
<code class="docutils literal notranslate"><span class="pre">datasets.HASYv2Dataset</span></code> class. However, it also can’t be uploaded here
because it exceeds 100MB. If you wish to use the codes presented here, you
need to unzip the dataset (which can be found
<a class="reference external" href="https://zenodo.org/record/259444#.YYwmp73MLUJ">here</a>) in the
<code class="docutils literal notranslate"><span class="pre">_Data</span></code> folder (creating a <code class="docutils literal notranslate"><span class="pre">HASYv2</span></code> folder with all of its contents).</p>
</div>
</div>
<div class="section" id="current-state-of-development">
<h2>Current State of Development<a class="headerlink" href="#current-state-of-development" title="Permalink to this headline">¶</a></h2>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>With the return of my classes, development will be slowed down
significantly in the next few months.</p>
</div>
<p>Base classes and a few models have already been defined, but development is
hindered by the time it takes to train models. Currently, using Google Colab’s
GPU acceleration, training on one fold takes several hours to complete (which
basically uses up the daily available GPU runtime). Training a model with all
10 folds would thus take up to 10 days for each model. Development should thus
shift to implementations with
<a class="reference external" href="https://www.pytorchlightning.ai/">PyTorch Lightning</a> and
<a class="reference external" href="https://github.com/pytorch/xla/">PyTorch/XLA</a> (see <a class="reference internal" href="#objectives"><span class="std std-ref">Objectives</span></a>),
that could allow for multicore TPU training in Colab, speeding up the process.</p>
<p>In terms of the Pygame implementation (see “pygame-tests” branch), much has yet
to be done and improved, but the base window is about what I had in mind. I have
understood a little bit better how Pygame works, which will help in the next
steps. The text box is still very limited, and I’ll be working on it in the
future.</p>
<div class="figure align-center" id="id3">
<a class="reference internal image-reference" href="_images/drawingboard.png"><img alt="*The current state of the Pygame UI implementation.*" src="_images/drawingboard.png" style="width: 640px;" /></a>
<p class="caption"><span class="caption-text"><em>The current state of the Pygame UI implementation.</em></span><a class="headerlink" href="#id3" title="Permalink to this image">¶</a></p>
</div>
</div>
<div class="section" id="objectives">
<h2>Objectives<a class="headerlink" href="#objectives" title="Permalink to this headline">¶</a></h2>
<p>Here’s list of a few objectives I had in mind when starting this project. It
contains some things I have completed and others that I still want to complete.</p>
<ul>
<li><p>[X] Understand decorators.</p>
<blockquote>
<div><p>Although I understand how they work and how to implement them, I haven’t yet
found much use. Yet.</p>
</div></blockquote>
</li>
<li><p>[X] Understand context managers.</p>
<blockquote>
<div><p>Not only have I understood <em>how</em> they work, I’ve developed the
<code class="docutils literal notranslate"><span class="pre">main.ReportManager</span></code> class specifically to deal with creating model
reports, something I already used to do in a more manual way before.</p>
</div></blockquote>
</li>
<li><p>[X] Switch to
<a class="reference external" href="https://google.github.io/styleguide/pyguide.html">Google’s Style</a></p>
<blockquote>
<div><p>Working on it!</p>
</div></blockquote>
</li>
<li><p>[X] (WIP) Write a complete documentation with Sphinx.</p>
<blockquote>
<div><p>I have already worked with Sphinx in the past and personally loved it.
This is a permanent work in progress, of course, but I’m currently testing a
new theme (<a class="reference external" href="https://github.com/pradyunsg/furo">Furo</a>) and haven’t yet
written a docstring for everything so it’s particularly empty as of know.
To access the documentation, start from the
<code class="docutils literal notranslate"><span class="pre">/_Sphinx/_build.html/index.html</span></code> file.</p>
</div></blockquote>
</li>
<li><p>[ ] (WIP) Implement an interface for real-time drawing and prediction.</p>
<blockquote>
<div><p>Development has started using the Pygame module.</p>
</div></blockquote>
</li>
<li><p>[ ] Try to use <a class="reference external" href="https://www.pytorchlightning.ai/">PyTorch Lightning</a> and
<a class="reference external" href="https://github.com/pytorch/xla/">PyTorch/XLA</a> for accelerating training
using cloud multi-core TPUs (in GoogleColab).</p>
<blockquote>
<div><p>Despite knowing how to use GoogleColab’s GPUs for accelerating PyTorch code,
TPUs and specifically multi-core parallelism is something I don’t (yet) know
how to work with.</p>
</div></blockquote>
</li>
<li><p>[ ] Perhaps learn and use <a class="reference external" href="https://optuna.org/">Optuna</a> for selecting training
and Neural Networks hyperparameters.</p></li>
<li><p>[ ] Develop more CNNs for testing.</p></li>
<li><p>[ ] Finish developing functions for evaluating trained model’s performances on the
HASYv2 dataset.</p>
<blockquote>
<div><p>Using the same parameters as the ones used in the
<a class="reference external" href="https://arxiv.org/abs/1701.08380">article</a>.</p>
</div></blockquote>
</li>
</ul>
<div class="toctree-wrapper compound">
<p class="caption"><span class="caption-text">Contents:</span></p>
<ul>
<li class="toctree-l1"><a class="reference internal" href="defs.html">Base Classes and Functions</a></li>
<li class="toctree-l1"><a class="reference internal" href="datasets.html">Dataset Classes</a></li>
<li class="toctree-l1"><a class="reference internal" href="nets.html">Neural Networks</a></li>
<li class="toctree-l1"><a class="reference internal" href="decorators.html">Decorators</a></li>
</ul>
</div>
</div>
</div>
<div class="section" id="indices-and-tables">
<h1>Indices and tables<a class="headerlink" href="#indices-and-tables" title="Permalink to this headline">¶</a></h1>
<ul class="simple">
<li><p><a class="reference internal" href="genindex.html"><span class="std std-ref">Index</span></a></p></li>
<li><p><a class="reference internal" href="py-modindex.html"><span class="std std-ref">Module Index</span></a></p></li>
<li><p><a class="reference internal" href="search.html"><span class="std std-ref">Search Page</span></a></p></li>
</ul>
</div>


              </div>
              
              
              <div class='prev-next-bottom'>
                
    <a class='right-next' id="next-link" href="defs.html" title="next page">Base Classes and Functions</a>

              </div>
              
          </main>
          

      </div>
    </div>
  
  <script src="_static/js/index.1c5a1a01449ed65a7b51.js"></script>

  <footer class="footer mt-5 mt-md-0">
  <div class="container">
    
    <div class="footer-item">
      <p class="copyright">
    &copy; Copyright 2021, Antonio Rocha.<br/>
</p>
    </div>
    
    <div class="footer-item">
      <p class="sphinx-version">
Created using <a href="http://sphinx-doc.org/">Sphinx</a> 4.0.2.<br/>
</p>
    </div>
    
  </div>
</footer>
  </body>
</html>