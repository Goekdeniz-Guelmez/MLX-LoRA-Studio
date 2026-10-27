<section class="hero">
  <div class="hero__copy">
    <h1 class="hero__title">Make the model<br><span>fit the work.</span></h1>
    <p class="hero__lede">A native Mac studio for adapting open language models. Choose a training method, understand the tradeoffs, and watch each run on the machine in front of you.</p>
    <div class="hero__actions">
      <a class="button button--primary" href="#/Train">Explore the workflow</a>
      <a class="button button--quiet" href="#/Algorithm-Guide">Find your method</a>
    </div>
    <p class="hero__note">On-device training · Open methods · No rented GPU required for compatible runs</p>
  </div>
  <figure class="hero__visual">
    <div class="hero-image-frame">
      <img src="https://raw.githubusercontent.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/main/Sources/Media/app_live_metrics.png" alt="MLX LoRA Studio showing a live training loss chart, throughput, memory use, and run metrics" fetchpriority="high">
    </div>
    <figcaption>A run stays visible: loss, throughput, memory, learning rate, and the log behind them.</figcaption>
  </figure>
</section>

<section class="home-section">
  <div class="home-section__intro">
    <h2 class="home-section__title">Good models should be within reach.</h2>
    <p>When a model needs to learn a particular voice, format, or task, the path to a useful adapter can be local and understandable. MLX LoRA Studio brings the training controls, resource checks, and run history together in a native Mac app.</p>
  </div>
  <div class="principle-grid">
    <article class="principle">
      <h3>Keep training close</h3>
      <p>For supported models, run fine-tuning on Apple Silicon you already have. Avoid a per-hour cloud GPU bill for every experiment and keep the training loop under your control.</p>
    </article>
    <article class="principle">
      <h3>Start with less memory</h3>
      <p>LoRA and DoRA update compact adapters. QLoRA can load the base model at lower precision. A live estimate and ResourceGuard help you choose a run that fits your Mac.</p>
    </article>
    <article class="principle">
      <h3>Keep data on your Mac</h3>
      <p>Training runs locally. Model and dataset downloads use the network; uploading an adapter to Hugging Face is a separate action you choose when you are ready.</p>
    </article>
  </div>
</section>

<section class="home-section">
  <div class="home-section__intro">
    <h2 class="home-section__title">See what the training is doing.</h2>
    <p>Make a run, follow its behavior, and keep the files needed to reproduce it. The interface is approachable on day one and leaves the underlying configuration inspectable when you want more control.</p>
  </div>
  <div class="workflow-grid">
    <article class="workflow">
      <h3>Choose the job</h3>
      <p>Select a local or Hugging Face model, a dataset, a training objective, and an adaptation method. The app exposes algorithm-specific settings alongside shared optimization controls.</p>
    </article>
    <article class="workflow">
      <h3>Watch the run</h3>
      <p>Follow loss, validation, learning rate, speed, token throughput, memory, and live logs. Pause, resume, or stop from the toolbar as your experiment requires.</p>
    </article>
    <article class="workflow">
      <h3>Keep the result</h3>
      <p>Runs retain their configuration, logs, status, and adapter checkpoints. Continue from an adapter, inspect the output in Finder, or start an optional Hub upload.</p>
    </article>
  </div>
  <figure class="showcase">
    <div class="showcase-image">
      <img src="https://raw.githubusercontent.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/main/Sources/Media/app_train_active.png" alt="MLX LoRA Studio training form with model, dataset, adapter settings, resource estimate, and a live run console" loading="lazy">
    </div>
    <figcaption>The training form keeps model, data, method, adapter settings, and run output together. Resource use varies with the model, sequence length, method, and Mac.</figcaption>
  </figure>
</section>

<section class="home-section">
  <div class="home-section__intro">
    <h2 class="home-section__title">The best algorithm is the one your data can support.</h2>
    <p>Start with the signal you have. Use demonstrations when you know what a good answer looks like. Use preference pairs when you can compare answers. Use reinforcement learning when a reliable reward can score generated outputs.</p>
  </div>
  <div class="method-paths">
    <a class="method-path" href="#/SFT">
      <strong>I can show the model what to do</strong>
      <span>Begin with Supervised Fine-Tuning. NLL is the baseline; Chunked NLL helps memory-bound long sequences, while Dynamic Fine-Tuning changes how token errors are weighted.</span>
    </a>
    <a class="method-path" href="#/DPO">
      <strong>I can tell which answer is better</strong>
      <span>Use DPO for curated chosen and rejected responses. Compare it with reference-free CPO or single-stage ORPO; add DSLA only when latent alignment is part of the experiment.</span>
    </a>
    <a class="method-path" href="#/GRPO">
      <strong>I can score what the model generates</strong>
      <span>Explore GRPO, KLPO, or an online preference loop when reward quality is measurable. More sampling means more compute and more ways for a weak reward to mislead training.</span>
    </a>
  </div>
  <p><a href="#/Algorithm-Guide">Read the field guide to all 12 training algorithms →</a></p>
</section>

<section class="home-section">
  <div class="home-section__intro">
    <h2 class="home-section__title">From a clear explanation to a runnable notebook.</h2>
    <p>Every method has a deeper reference covering its objective, dataset shape, controls, and common failure modes. The app uses the open-source <a href="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora">mlx-lm-lora Python package</a> to run training. For a code-first route, open the <a href="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora-example-notebooks">notebook examples</a> or browse the <a href="https://github.com/Goekdeniz-Guelmez">GitHub profile</a>.</p>
  </div>
  <div class="audience-grid">
    <article class="audience">
      <h3>First experiment</h3>
      <p>Learn the fine-tuning vocabulary, choose a small instruct model, and start with a short SFT run you can evaluate yourself.</p>
    </article>
    <article class="audience">
      <h3>Research and engineering</h3>
      <p>Compare preference objectives, inspect the YAML run configuration, read the logs, and re-run a saved experiment from the command line.</p>
    </article>
    <article class="audience">
      <h3>Teams exploring local AI</h3>
      <p>Prototype a task-specific adapter on Apple hardware, with local training artifacts and an optional, deliberate publishing step.</p>
    </article>
  </div>
  <p><a href="#/Notebooks">Open the example notebooks →</a></p>
</section>

<section class="quote-panel">
  <h2>Make private model adaptation usable.</h2>
  <div>
    <p>Useful AI will be shaped for specific work. MLX LoRA Studio brings that adaptation into a native Mac workflow, backed by the mlx-lm-lora Python trainer, runnable notebooks, and MCP access for agent workflows. Sponsorship and strategic partnerships can help improve model coverage, reliability, and the path from a base model to a useful adapter.</p>
    <div class="quote-actions">
      <a class="button button--primary" href="https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio" target="_blank" rel="noopener">Explore the project</a>
      <a class="button button--quiet" href="https://github.com/sponsors/Goekdeniz-Guelmez" target="_blank" rel="noopener">Sponsor the work</a>
      <a class="button button--quiet" href="https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/issues/new" target="_blank" rel="noopener">Discuss a partnership</a>
    </div>
  </div>
</section>

<footer class="page-footer">
  <span>MLX LoRA Studio · macOS 14+ · Apple Silicon</span>
  <span><a href="https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio">Open-source project</a> · <a href="MCP-Server">Agent and MCP workflow</a></span>
</footer>
