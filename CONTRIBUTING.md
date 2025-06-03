# CONTRIBUTING

## 1. Welcome & Scope

`saev` is research code.
PRs that fix bugs, add datasets, or improve docs are welcome.
Large architectural rewrites: please open a discussion first.

## 2. TL;DR

Install [uv](https://docs.astral.sh/uv/).
Clone this repository, then from the root directory:

```sh
uv run python -m saev --help
```

You also need [yek](https://github.com/bodo-run/yek) and [lychee](https://github.com/lycheeverse/lychee) for generating docs.

If you want to do any of the web interface work, you need [elm](https://guide.elm-lang.org/install/elm.html), [elm-format](https://github.com/avh4/elm-format/releases/latest) and [tailwindcss](https://github.com/tailwindlabs/tailwindcss/releases/latest).

## 3. Repo Layout & Data Dependencies

```
src/
  saev/
    __main__.py  <- entrypoint for entire package.
    config.py    <- All configs, which define the CLI flags.

    activations.py  <- For recording, storing and loading transformer activations.
    training.py     <- For training sweeps of SAEs on transformer activations.
    visuals.py      <- For making lots of images

    nn/
      modeling.py    <- Activations functions for SAEs (JumpReLU, TopK).
      objectives.py  <- Objective functions for SAEs.

  docs/
    templates/  <- pdoc3 templates

  web/
    apps/  <- HTML/CSS and JS (compiled Elm)
    src/   <- Elm source code
```

## 4. Coding Style & Conventions

* Don't hard-wrap comments. Only use linebreaks for new paragraphs. Let the editor soft wrap content.
* Use single-backticks for variables. We use Markdown and [pdoc3](https://pdoc3.github.io/pdoc/) for docs rather than ReST and Sphinx.
* File descriptors from `open()` are called `fd`.
* Use types where possible, including `jaxtyping` hints.
* Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
* Variables referring to a filepath should be suffixed with `_fpath`. Directories are `_dpath`.
* Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
* Only use `setup` for naming functions that don't return anything.

Throughout the code, variables are annotated with shape suffixes, as [recommended by Noam Shazeer](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).

The key for these suffixes:

* B: batch size
* W: width in patches (typically 14 or 16)
* H: height in patches (typically 14 or 16)
* D: ViT activation dimension (typically 768 or 1024)
* S: SAE latent dimension (768 x 16, etc)
* L: Number of latents being manipulated at once (typically 1-5 at a time)
* C: Number of classes in ADE20K (151)

For example, an activation tensor with shape (batch, width, height d_vit) is `acts_BWHD`.

## 5. Testing & Linting

`justfile` contains commands for testing and linting.

`just lint` will format and lint.
`just test` will format, lint and test, then report coverage.

To run just one test, run `uv run python -m pytest src/saev -k TESTNAME`.

## 6. Commit / PR Checklist

1. Run `just test`.
2. Check that there are no regressions. Unless you are certain tests are not needed, the coverage % should either stay the same or increase.
3. Run `just docs`.
4. Fix any missing doc links.

## 7. Research Reproducibility Notes

If you add a new neural network or other hard-to-unit-test bit of code, it should either be a trivial change or it should come with an experiment demonstrating that it works.

This means some links to WandB, or a small report in markdown in the repo itself.
For example, if you wanted to add a new activation function from a recent paper, you should train a small sweep using the current baseline, demonstrate some qualitative or quantitative results, and then run the same sweep with your minimal change, and demonstrate some improvement (speed, quality, loss, etc).
Document this in a markdown report (in `src/saev/nn` for a new activation function) and include it in the docs.

Neural networks are hard. It's okay.

## 8. Code of Conduct & License Footnotes

Be polite, kind and assume good intent.
